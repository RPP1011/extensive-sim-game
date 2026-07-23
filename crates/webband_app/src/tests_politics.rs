//! S11 in-crate tests: THE GUILD LAYER — factions, petitions, the founders'
//! ambition, bands, afield, markets/knowledge.
//!
//! Every numeric pin is DERIVED from the TS source (`F:\MB\src\guild\*.ts`),
//! cited at the assertion, never guessed. Every LAW named in a module header
//! has a test here that fails if it is broken.

use crate::afield::{
    afield_phase, dispatch_cost, dispatch_party, is_afield, larder_nutrition, pack_rations,
    sync_afield, AfieldOutcome, AfieldPhase, DispatchContext, DispatchOpts, Errand, ErrandFight,
};
use crate::ambition::{roll_ambition, AmbitionStep, AmbitionTruth, StageKind};
use crate::bands::{
    band_members, can_recruit, need_wealth, request_cause, sign_band, sign_price, tick_band_goals,
    BandTickInput, SignHow,
};
use crate::campaign::{
    build_epilogue, check_ambition, dawn_fold, dawn_fold_political, founding_snapshot,
    petition_capacity, Campaign, ColonySnapshot, MemberView,
};
use crate::defs::{BuildingView, InventorySnapshot, StackView};
use crate::director::{tick_director_full, DirectorView, TropeKind};
use crate::factions::{
    faction_holding, petitioners, wild_power, FactionKind, FACTION_KINDS,
};
use crate::founding::new_founding;
use crate::knowledge::threat_strength_word;
use crate::markets::{
    believed_market, haulage_from, market_at, market_epoch, market_known, market_kind_table,
    nearest_market, provision_choice, MARKET_SEASON_DAYS,
};
use crate::petitions::{
    answer_petition, lapse_petitions, petition_choices, petition_pay, petition_spec,
    resolve_sent_petition, standing_with, standing_word, Petition, PetitionCapacity,
    PetitionChoiceKind, PetitionKind, StandingLedger, PETITIONS, PETITION_DAYS,
};
use crate::rng::RngState;
use crate::scenario::{ScenarioId, ALL_SCENARIOS};
use crate::worldgen::{BandGoal, BandStatus};

fn political(seed: u32, scenario: ScenarioId) -> Campaign {
    Campaign::new_political(new_founding(seed, 0, scenario).expect("founding"))
}

/// The first seed in range whose country satisfies `want`.
fn seed_where(want: impl Fn(&Campaign) -> bool) -> Campaign {
    for seed in 1u32..400 {
        let c = political(seed.wrapping_mul(2_654_435_761), ScenarioId::Village);
        if want(&c) {
            return c;
        }
    }
    panic!("no seed in 400 tries produced the country this test needs");
}

fn snap_of(c: &Campaign) -> ColonySnapshot {
    founding_snapshot(c, 60.0)
}

fn patience_of(c: &Campaign, band: &str) -> i64 {
    c.band_states.iter().find(|(id, _)| id == band).unwrap().1.state.patience
}

fn status_of(c: &Campaign, band: &str) -> BandStatus {
    c.band_states.iter().find(|(id, _)| id == band).unwrap().1.state.status
}

// ===========================================================================
// FACTIONS
// ===========================================================================

/// The TS validates its kind table at module load (`factions.ts:66-70`).
#[test]
fn faction_kind_table_is_valid() {
    for s in &FACTION_KINDS {
        assert!(!s.seats.is_empty(), "faction kind needs seats: {:?}", s.kind);
        assert!(!s.wants.is_empty(), "faction kind needs prose: {:?}", s.kind);
    }
    // The wild power never petitions — it only threatens.
    assert!(!FACTION_KINDS.iter().find(|s| s.kind == FactionKind::Wild).unwrap().petitions);
    assert_eq!(FACTION_KINDS.iter().filter(|s| s.petitions).count(), 3);
    // The draw ORDER is the table order.
    assert_eq!(
        FACTION_KINDS.map(|s| s.kind),
        [FactionKind::Crown, FactionKind::Church, FactionKind::Mercantile, FactionKind::Wild]
    );
}

#[test]
fn powers_are_generated_from_the_landmarks_the_country_rolled() {
    let mut seen_kinds = std::collections::BTreeSet::new();
    for seed in 1u32..60 {
        let c = political(seed.wrapping_mul(2_654_435_761), ScenarioId::Village);
        let lms = &c.founding.world.landmarks;
        assert!(!c.factions.is_empty(), "a country with landmarks seats at least one power");
        assert!(c.factions.len() <= 4, "at most one power per kind (+ the lone-power rival)");

        let mut seats: Vec<&str> = Vec::new();
        for f in &c.factions {
            seen_kinds.insert(f.kind);
            let seat = c.founding.world.landmark_by_id(&f.seat_landmark_id).expect("seat exists");
            // The lone-power rival is the one faction seated off its own table.
            let is_rival = c.factions.len() == 2
                && f.kind == FactionKind::Wild
                && !crate::factions::faction_kind_spec(FactionKind::Wild).seats.contains(&seat.kind);
            if !is_rival {
                assert!(
                    crate::factions::faction_kind_spec(f.kind).seats.contains(&seat.kind),
                    "{:?} seated at a {:?}",
                    f.kind,
                    seat.kind
                );
            }
            assert!(!seats.contains(&f.seat_landmark_id.as_str()), "two powers on one seat");
            seats.push(&f.seat_landmark_id);
        }
        // EVERY place is held, and held ONCE — the nearest-seat claim.
        for lm in lms {
            let holders: Vec<&str> = c
                .factions
                .iter()
                .filter(|f| f.hold_ids.contains(&lm.id))
                .map(|f| f.id.as_str())
                .collect();
            assert_eq!(holders.len(), 1, "{} held by {holders:?}", lm.name);
        }
        assert_eq!(
            c.factions.iter().map(|f| f.hold_ids.len()).sum::<usize>(),
            lms.len(),
            "holds partition the country"
        );
        // A country with politics has more than one voice.
        assert!(c.factions.len() >= 2 || lms.len() <= 1);
    }
    // Across 60 countries every kind of power appears somewhere.
    assert_eq!(seen_kinds.len(), 4, "all four kinds are reachable: {seen_kinds:?}");
}

/// LAW: holds are PERSISTED, never re-derived.
#[test]
fn holds_are_persisted_never_rederived() {
    let c = political(0xC0_FFEE, ScenarioId::Village);
    let before: Vec<Vec<String>> = c.factions.iter().map(|f| f.hold_ids.clone()).collect();
    let json = serde_json::to_string(&c).expect("serialize");
    let back: Campaign = serde_json::from_str(&json).expect("deserialize");
    let after: Vec<Vec<String>> = back.factions.iter().map(|f| f.hold_ids.clone()).collect();
    assert_eq!(before, after);
    // And the accessor reads the STORED list, not a distance computation:
    for f in &back.factions {
        for h in &f.hold_ids {
            assert_eq!(
                faction_holding(&back.factions, h).map(|x| x.id.as_str()),
                Some(f.id.as_str())
            );
        }
    }
}

/// LAW: the wild power AUTHORS RAIDS — a raid becomes a chapter of one
/// running quarrel rather than weather.
#[test]
fn the_wild_power_authors_raids() {
    let mut c = seed_where(|c| wild_power(&c.factions).is_some());
    let wild = wild_power(&c.factions).unwrap().id.clone();
    c.director.points = 500;
    c.director.plan = Some(TropeKind::Raid);
    c.day = 30;
    let view = DirectorView { wealth: 500, avg_mood: 60.0, ..Default::default() };
    let tick = tick_director_full(&mut c, &view);
    let ev = tick.event.expect("the committed raid fires");
    assert_eq!(ev.kind, TropeKind::Raid);
    let raid = c.raid.as_ref().expect("raid staged");
    assert_eq!(raid.faction_id.as_deref(), Some(wild.as_str()));
    assert!(ev.text.contains("has stirred"), "the author is named: {}", ev.text);
}

/// LAW: hostility is a LATCH with EXACTLY TWO doors out — tribute paid, or
/// their raid beaten. Never time.
#[test]
fn hostility_latches_and_has_exactly_two_doors() {
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty());
    let fid = petitioners(&c.factions)[0].id.clone();
    c.faction_ledger.at(&fid).hostile_since = Some(1);

    // DOOR ZERO — time — is not a door.
    for _ in 0..200 {
        let snap = snap_of(&c);
        let _ = dawn_fold(&mut c, &snap, &[]);
        assert!(
            c.faction_ledger.is_hostile(&fid),
            "time cleared a latched hostility on day {}",
            c.day
        );
        c.raid = None; // keep the storyteller speaking; the latch is the subject
    }

    // DOOR ONE: pay their tribute.
    c.gold = 5000;
    stage_petition_for(&mut c, &fid, PetitionKind::Tribute, 0);
    answer_petition(&mut c, PetitionChoiceKind::Pay, PetitionCapacity::default())
        .expect("tribute affordable");
    assert!(!c.faction_ledger.is_hostile(&fid), "tribute is door one");

    // DOOR TWO: beat the raid they sent. (The price a POOR guild pays —
    // without it a broke colony that angered a power could never recover.)
    c.faction_ledger.at(&fid).hostile_since = Some(c.day);
    c.gold = 0;
    let raid = crate::raids::spawn_raid(
        &mut c.rng,
        c.day,
        200,
        c.roster.len(),
        c.founding.scenario,
        crate::raids::SpawnOpts { faction_id: Some(fid.clone()), ..Default::default() },
    );
    c.raid = Some(raid.clone());
    let result = crate::campaign::RaidResultView {
        victory: true,
        gold_looted: 10,
        member_hp: c.roster.iter().map(|id| (id.clone(), 1.0)).collect(),
    };
    crate::campaign::resolve_raid(&mut c, &raid, &result, &InventorySnapshot::default());
    assert!(!c.faction_ledger.is_hostile(&fid), "blood is door two");
}

/// LAW (structural): a faction has no beliefs, it has a LEDGER. The module may
/// not reach into the minds vocabulary, and standing — which decays and is
/// therefore belief-shaped — deliberately lives elsewhere.
#[test]
fn factions_hold_a_ledger_never_beliefs() {
    // Scan the CODE, not the module's own prose about the law.
    let src: String = include_str!("factions.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("
")
        .to_lowercase();
    for banned in ["minds", "belief", "grudge", "gossip", "thought", "standing"] {
        assert!(
            !src.contains(banned),
            "factions.rs code mentions `{banned}` — the epistemic split forbids it"
        );
    }
    // FactionState is four plain counters and nothing else.
    let st = crate::factions::FactionState::default();
    assert_eq!((st.served, st.refused, st.last_petition_day), (0, 0, -99));
    assert!(st.hostile_since.is_none());
}

// ===========================================================================
// THE FOUNDING DRAW ORDER
// ===========================================================================

/// THE APPEND CLAIM, VERIFIED: rolling factions + ambition changes NOTHING
/// about the founding rolls — but it DOES move the stream position every
/// post-founding draw resumes from, which is exactly why it is a second
/// constructor rather than a change to `new_founding`.
#[test]
fn politics_roll_is_append_only() {
    for seed in [1u32, 7, 4242, 0xDEAD_BEEF] {
        let f = new_founding(seed, 0, ScenarioId::Village).expect("founding");
        let plain = Campaign::new(f.clone());
        let politic = Campaign::new_political(f.clone());

        // Every founding roll is byte-identical.
        assert_eq!(plain.founding.cast, politic.founding.cast);
        assert_eq!(plain.founding.world, politic.founding.world);
        assert_eq!(plain.founding.band_states, politic.founding.band_states);
        assert_eq!(plain.founding.colony_terrain, politic.founding.colony_terrain);
        assert_eq!(plain.founding.rng_counter, politic.founding.rng_counter);
        assert_eq!(plain.roster, politic.roster);

        // The stream MOVED — appending is not free downstream.
        assert!(
            politic.rng.rng_counter > plain.rng.rng_counter,
            "the politics roll must consume draws"
        );
        // …and the unpolitical campaign is untouched by this slice.
        assert_eq!(plain.rng.rng_counter, f.rng_counter);
        assert!(plain.factions.is_empty() && plain.ambition.is_none());
        assert!(!plain.politics_enabled && politic.politics_enabled);
    }
}

/// THE COMPARABILITY LAW still holds WITH factions and ambition rolling: one
/// seed, four starting conditions, the same country and the same powers.
#[test]
fn comparability_holds_with_politics_rolled() {
    let seed = 0x5EED_1234;
    let base = political(seed, ALL_SCENARIOS[0]);
    for s in ALL_SCENARIOS {
        let c = political(seed, s);
        assert_eq!(c.founding.cast, base.founding.cast, "{s:?} rolled a different cast");
        assert_eq!(c.founding.world, base.founding.world, "{s:?} rolled a different country");
        assert_eq!(c.factions, base.factions, "{s:?} rolled different powers");
        assert_eq!(
            c.ambition.as_ref().map(|a| (a.title.clone(), a.stages.len())),
            base.ambition.as_ref().map(|a| (a.title.clone(), a.stages.len())),
            "{s:?} rolled a different arc"
        );
        assert_eq!(c.rng.rng_counter, base.rng.rng_counter, "{s:?} spent different draws");
        // The scenario's standing offset is the ONLY politics difference, and
        // it takes no draws.
        let offset = crate::scenario::scenario_spec(s).standing;
        for f in petitioners(&c.factions) {
            assert!((standing_with(&c, &f.id) - offset as f64).abs() < 1e-9);
        }
    }
}

/// DIGEST SAFETY: a campaign founded the old way serializes without a single
/// new key, so the S6 soak's `fnv1a(serde_json(campaign))` pin cannot move.
#[test]
fn an_unpolitical_campaign_serializes_without_the_new_fields() {
    let c = Campaign::new(new_founding(4242, 0, ScenarioId::Village).expect("founding"));
    let json = serde_json::to_string(&c).expect("serialize");
    for key in [
        "politics_enabled",
        "\"factions\"",
        "faction_ledger",
        "standing_ledger",
        "\"petition\"",
        "ambition",
        "afield",
        "market_intel",
        "provision_source",
        "sacked",
        "satisfied_day",
        "deferred_by_coin",
        "want_met_day",
    ] {
        assert!(!json.contains(key), "an unpolitical save leaked `{key}`");
    }
}

/// S13 — THE ASK IS THE GATE, and a campaign gets more than one of them.
///
/// `c.petition_open` was set when a petition opened and cleared by nothing: all
/// four answers (send-home / pay / refuse / lapse) drop `c.petition` and left
/// the flag standing, so the storyteller's petition trope was permanently
/// ineligible after the FIRST ask. Nothing caught it because no campaign in the
/// port had ever answered one and kept playing. This drives 90 political days,
/// answering every ask by refusing it (the free answer, so the run never
/// depends on the purse), and demands the country keep asking.
#[test]
fn petitions_keep_coming_after_the_first_is_answered() {
    let mut c = political(20260722, ScenarioId::Village);
    let mut opened = 0usize;
    let mut answered = 0usize;
    for _ in 0..120 {
        let snap = snap_of(&c);
        let out = dawn_fold(&mut c, &snap, &[]);
        if out.petition_opened.is_some() {
            opened += 1;
        }
        // A raid the caller never settles freezes the storyteller forever
        // ("one storm at a time"), so settle each one as the bridge does.
        if let Some(r) = &c.raid.clone() {
            let result = crate::campaign::RaidResultView {
                victory: true,
                gold_looted: 10,
                member_hp: c.roster.iter().map(|id| (id.clone(), 0.9)).collect(),
            };
            crate::campaign::resolve_raid(&mut c, r, &result, &snap.inventory);
        }
        // Answer it the day it lands — refusing costs only standing.
        if c.petition.is_some() {
            let cap = petition_capacity(&c, &snap);
            if answer_petition(&mut c, PetitionChoiceKind::Refuse, cap).is_some() {
                answered += 1;
                assert!(c.petition.is_none(), "an answered ask must close");
                assert!(!c.petition_open, "…and the flag must close with it");
            }
        }
    }
    println!("[S13 gate] 120 political days opened {opened} petitions");
    assert!(
        opened >= 3,
        "120 political days produced {opened} petitions — the gate is stuck again"
    );
    assert_eq!(opened, answered, "every ask that opened was answerable");
}

/// DIGEST SAFETY, the behavioural half: with politics OFF, a long fold never
/// grows one of the new fields and never re-words a line the chronicle already
/// had. (The S6 soak hashes `serde_json(campaign)`, so a single changed
/// character in a chronicle entry moves its pin.)
#[test]
fn an_unpolitical_campaign_stays_pre_s11_through_a_long_fold() {
    let mut c = Campaign::new(new_founding(20260722, 0, ScenarioId::Village).expect("founding"));
    for _ in 0..120 {
        let snap = snap_of(&c);
        let out = dawn_fold(&mut c, &snap, &[]);
        // The politics steps must all be inert.
        assert!(out.petition_opened.is_none() && out.petition_lapsed.is_none());
        assert!(out.ambition.is_none() && out.epilogue.is_none());
        assert_eq!(out.bands, Default::default());
        assert!(out.thoughts.is_empty());
        // …and none of the new state ever fills.
        assert!(c.factions.is_empty() && c.faction_ledger.is_empty());
        assert!(c.standing_ledger.is_empty() && c.afield.is_empty());
        assert!(c.petition.is_none() && c.ambition.is_none());
        assert!(c.market_intel.is_empty() && c.sacked.is_empty());
        if let Some(r) = &c.raid.clone() {
            let result = crate::campaign::RaidResultView {
                victory: c.day % 2 == 0,
                gold_looted: 10,
                member_hp: c.roster.iter().map(|id| (id.clone(), 0.9)).collect(),
            };
            crate::campaign::resolve_raid(&mut c, r, &result, &snap.inventory);
        }
    }
    let json = serde_json::to_string(&c).expect("serialize");
    for key in ["politics_enabled", "\"factions\"", "faction_ledger", "standing_ledger",
                "\"petition\"", "ambition", "afield", "market_intel", "sacked",
                "satisfied_day", "deferred_by_coin", "want_met_day"] {
        assert!(!json.contains(key), "120 folds leaked `{key}` into an unpolitical save");
    }
}

/// The AUTHORLESS raid prose is byte-for-byte what it was before the wild
/// power could author one — the chronicle line the soak's digest hashes.
#[test]
fn an_authorless_raid_reads_exactly_as_it_did() {
    let mut c = Campaign::new(new_founding(4242, 0, ScenarioId::Village).expect("founding"));
    c.day = 30;
    c.director.points = 500;
    c.director.plan = Some(TropeKind::Raid);
    let ev = tick_director_full(
        &mut c,
        &DirectorView { wealth: 500, avg_mood: 60.0, ..Default::default() },
    )
    .event
    .expect("the raid fires");
    let warband = c
        .raid
        .as_ref()
        .unwrap()
        .elite_name
        .as_ref()
        .map_or("a warband".to_string(), |n| format!("{n}'s warband"));
    assert_eq!(
        ev.text,
        format!("Riders were seen circling the valley — {warband} will be at the fences by dawn. One night to make ready.")
    );
    assert!(c.raid.as_ref().unwrap().faction_id.is_none());
}

// ===========================================================================
// PETITIONS
// ===========================================================================

#[test]
fn petition_table_is_valid() {
    for s in &PETITIONS {
        assert!(s.weight > 0.0, "petition weight must be positive: {:?}", s.kind);
        assert!(s.hands >= 0.0 && s.days >= 0 && s.pay_per_hand >= 0);
        assert!(!s.asks.is_empty() && !s.because.is_empty());
    }
    // The WAGE table (petitions.ts:50-72): levy 8 / escort 7 / relief 6 /
    // arbitration 10; tithe and tribute are extraction and pay NOTHING.
    assert_eq!(petition_spec(PetitionKind::Levy).pay_per_hand, 8);
    assert_eq!(petition_spec(PetitionKind::Escort).pay_per_hand, 7);
    assert_eq!(petition_spec(PetitionKind::Relief).pay_per_hand, 6);
    assert_eq!(petition_spec(PetitionKind::Arbitration).pay_per_hand, 10);
    assert_eq!(petition_spec(PetitionKind::Tithe).pay_per_hand, 0);
    assert_eq!(petition_spec(PetitionKind::Tribute).pay_per_hand, 0);
    // And the standing weights.
    assert_eq!(petition_spec(PetitionKind::Levy).weight, 12.0);
    assert_eq!(petition_spec(PetitionKind::Relief).weight, 14.0);
    assert_eq!(petition_spec(PetitionKind::Tribute).weight, 16.0);
    assert_eq!(PETITION_DAYS, 6);
}

/// LAW: standing drifts AT READ, asymmetrically — 0.5/day up from negative,
/// 0.25/day down from positive (`petitions.ts:101-110`). No per-day pass.
#[test]
fn standing_drift_is_lazy_and_asymmetric() {
    let mut l = StandingLedger::default();
    l.move_by(0, "f", -20.0);
    // Hatred fades at 0.5/day.
    assert!((l.get(10, "f") - -15.0).abs() < 1e-9);
    assert!((l.get(40, "f") - 0.0).abs() < 1e-9, "never overshoots past zero");
    // Love fades at HALF that.
    let mut l2 = StandingLedger::default();
    l2.move_by(0, "f", 20.0);
    assert!((l2.get(10, "f") - 17.5).abs() < 1e-9);
    assert!((l2.get(400, "f") - 0.0).abs() < 1e-9);
    // THE ASYMMETRY, stated as a ratio: being hated fades exactly twice as
    // fast as being liked. That constant IS the anti-death-spiral.
    let up = (l.get(10, "f") - l.get(0, "f")) / 10.0;
    let down = (l2.get(0, "f") - l2.get(10, "f")) / 10.0;
    assert!((up / down - 2.0).abs() < 1e-9, "up {up} down {down}");
    // Reading never mutates.
    assert!((l.get(10, "f") - -15.0).abs() < 1e-9);
    // Clamped to ±100.
    let mut l3 = StandingLedger::default();
    l3.move_by(0, "f", 500.0);
    assert!((l3.get(0, "f") - 100.0).abs() < 1e-9);
    l3.move_by(0, "f", -500.0);
    assert!((l3.get(0, "f") - -100.0).abs() < 1e-9);
    // The tier is a WORD, derived.
    assert_eq!(standing_word(70.0), "sworn friends");
    assert_eq!(standing_word(25.0), "well thought of");
    assert_eq!(standing_word(0.0), "known to them");
    assert_eq!(standing_word(-30.0), "out of favour");
    assert_eq!(standing_word(-80.0), "hated");
}

fn stage_petition_for(c: &mut Campaign, fid: &str, kind: PetitionKind, hands: u32) {
    let spec = petition_spec(kind);
    c.petition = Some(Petition {
        id: format!("p{}_{fid}", c.day),
        faction_id: fid.to_string(),
        kind,
        landmark_id: None,
        need_hands: hands,
        need_provisions: hands * spec.days as u32,
        need_gold: i64::from(hands) * spec.gold_per_hand,
        posted_day: c.day,
        expires_day: c.day + PETITION_DAYS,
        chosen: None,
        afield_id: None,
    });
}

fn stage_petition(c: &mut Campaign, kind: PetitionKind, hands: u32) -> String {
    let fid = petitioners(&c.factions)[0].id.clone();
    stage_petition_for(c, &fid, kind, hands);
    fid
}

/// LAW: LAPSING COSTS 1.5× A REFUSAL. Silence is contempt — that is what stops
/// the panel being dismissible (`petitions.ts:334`).
#[test]
fn lapsing_costs_one_and_a_half_times_refusing() {
    let refuser0 = seed_where(|c| petitioners(&c.factions).len() >= 2);
    let mut refuser = refuser0.clone();
    let mut lapser = refuser0.clone();
    let offset = crate::scenario::scenario_spec(refuser.founding.scenario).standing as f64;

    let fid = stage_petition(&mut refuser, PetitionKind::Relief, 3);
    answer_petition(&mut refuser, PetitionChoiceKind::Refuse, PetitionCapacity::default())
        .expect("refusing is always available");
    let refused_delta = standing_with(&refuser, &fid) - offset;

    stage_petition(&mut lapser, PetitionKind::Relief, 3);
    lapser.day += PETITION_DAYS; // the deadline falls due
    let lapse = lapse_petitions(&mut lapser).expect("the ask fell due");
    // Read at the SAME instant it was written so the lazy drift is 0.
    let lapsed_delta = lapser.standing_ledger.get(lapser.day, &fid)
        - (offset - PETITION_DAYS as f64 * crate::petitions::DOWN_PER_DAY).max(0.0);

    let w = petition_spec(PetitionKind::Relief).weight;
    assert!((refused_delta - -w).abs() < 1e-9, "a refusal costs the kind's weight ({w})");
    assert!((lapse.standing_cost - w * 1.5).abs() < 1e-9);
    assert!((lapsed_delta - -(w * 1.5)).abs() < 1e-9, "lapse delta {lapsed_delta}");
    assert!(
        (lapsed_delta / refused_delta - 1.5).abs() < 1e-9,
        "lapse {lapsed_delta} vs refuse {refused_delta}"
    );
    assert!(lapser.petition.is_none(), "a lapsed ask closes");
    // A lapse is still an insult on the ledger.
    assert_eq!(lapser.faction_ledger.get(&fid).refused, 1);
}

/// LAW: refusing PLEASES THEIR RIVALS — the field that turns a standing table
/// into politics (`petitions.ts:302-307`). The wild power asks nothing and is
/// indifferent.
#[test]
fn refusing_pleases_their_rivals() {
    let mut c =
        seed_where(|c| petitioners(&c.factions).len() >= 2 && wild_power(&c.factions).is_some());
    let fid = stage_petition(&mut c, PetitionKind::Levy, 3);
    let wild = wild_power(&c.factions).unwrap().id.clone();
    answer_petition(&mut c, PetitionChoiceKind::Refuse, PetitionCapacity::default())
        .expect("refuse");

    let w = petition_spec(PetitionKind::Levy).weight;
    let offset = crate::scenario::scenario_spec(c.founding.scenario).standing as f64;
    assert!((standing_with(&c, &fid) - (offset - w)).abs() < 1e-9);
    for f in petitioners(&c.factions) {
        if f.id == fid {
            continue;
        }
        assert!(
            (standing_with(&c, &f.id) - (offset + w * 0.25)).abs() < 1e-9,
            "rival {} was not pleased",
            f.name
        );
    }
    assert!(
        (standing_with(&c, &wild) - 0.0).abs() < 1e-9,
        "the wild power asks nothing and cares nothing"
    );
    assert_eq!(c.faction_ledger.get(&fid).refused, 1);
}

/// LAW: the choices are a SIM SEAM — each carries its cost and its blocked
/// reason, and a blocked choice cannot be taken (`canPlace`, for politics).
#[test]
fn petition_choices_are_verdicts_and_block_honestly() {
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty());
    c.gold = 0;
    stage_petition(&mut c, PetitionKind::Levy, 4);
    let p = c.petition.clone().unwrap();

    // No one to send, no coin to pay.
    let none = petition_choices(&c, &p, PetitionCapacity { available_hands: 0, meals: 0 });
    let send = none.iter().find(|o| o.choice == PetitionChoiceKind::Send).unwrap();
    assert_eq!(send.blocked.as_deref(), Some("there is no one here to send"));
    let pay = none.iter().find(|o| o.choice == PetitionChoiceKind::Pay).unwrap();
    assert_eq!(pay.blocked.as_deref(), Some("the coffer holds 0"));
    assert!(none
        .iter()
        .find(|o| o.choice == PetitionChoiceKind::Refuse)
        .unwrap()
        .blocked
        .is_none());
    assert!(
        answer_petition(
            &mut c,
            PetitionChoiceKind::Send,
            PetitionCapacity { available_hands: 0, meals: 0 }
        )
        .is_none(),
        "a blocked choice cannot be taken"
    );

    // Some, but not enough — the count is named.
    let few = petition_choices(&c, &p, PetitionCapacity { available_hands: 2, meals: 99 });
    let send = few.iter().find(|o| o.choice == PetitionChoiceKind::Send).unwrap();
    assert_eq!(
        send.blocked.as_deref(),
        Some("only 2 of 4 can ride — the rest are away, laid up, or will not stir")
    );

    // Hands enough, larder thin (levy days 3 × 4 hands = 12 rations).
    let dry = petition_choices(&c, &p, PetitionCapacity { available_hands: 9, meals: 3 });
    let send = dry.iter().find(|o| o.choice == PetitionChoiceKind::Send).unwrap();
    assert_eq!(send.blocked.as_deref(), Some("the larder cannot spare 12 — there is 3"));

    // Clear, and the wage is ADVERTISED on the send choice.
    let ok = petition_choices(&c, &p, PetitionCapacity { available_hands: 9, meals: 99 });
    let send = ok.iter().find(|o| o.choice == PetitionChoiceKind::Send).unwrap();
    assert!(send.blocked.is_none());
    assert!(send.detail.contains(&format!("{} gold when it is done", petition_pay(&p))));

    // A tribute has no send option at all (hands 0).
    stage_petition(&mut c, PetitionKind::Tribute, 0);
    let t = c.petition.clone().unwrap();
    let opts = petition_choices(&c, &t, PetitionCapacity { available_hands: 9, meals: 99 });
    assert!(opts.iter().all(|o| o.choice != PetitionChoiceKind::Send));
}

/// The four answers pay four different prices (`petitions.ts payOff` shares:
/// send-won 1, pay 0.5, send-failed 0.25, refuse −1, lapse −1.5).
#[test]
fn the_answers_pay_their_own_prices() {
    let base = seed_where(|c| !petitioners(&c.factions).is_empty());
    let w = petition_spec(PetitionKind::Levy).weight;
    let offset = crate::scenario::scenario_spec(base.founding.scenario).standing as f64;

    // PAY — half credit, and the coin leaves the chest.
    let mut c = base.clone();
    c.gold = 1000;
    let fid = stage_petition(&mut c, PetitionKind::Levy, 3);
    let owed = c.petition.as_ref().unwrap().need_gold;
    assert_eq!(owed, 3 * 60, "goldPerHand 60 × 3 hands");
    let renown0 = c.renown;
    answer_petition(&mut c, PetitionChoiceKind::Pay, PetitionCapacity::default()).unwrap();
    assert_eq!(c.gold, 1000 - owed);
    assert!((standing_with(&c, &fid) - (offset + w * 0.5)).abs() < 1e-9);
    assert_eq!(c.renown - renown0, crate::defs::js_round(w * 0.5 * 0.5));
    assert_eq!(c.faction_ledger.get(&fid).served, 1);

    // SEND, won — full credit AND the wage comes home.
    let mut c = base.clone();
    c.gold = 0;
    stage_petition(&mut c, PetitionKind::Levy, 3);
    let pay = petition_pay(c.petition.as_ref().unwrap());
    assert_eq!(pay, 8 * 3 * 3, "payPerHand 8 × 3 hands × 3 days");
    answer_petition(
        &mut c,
        PetitionChoiceKind::Send,
        PetitionCapacity { available_hands: 9, meals: 99 },
    )
    .unwrap();
    assert_eq!(c.petition.as_ref().unwrap().chosen, Some(PetitionChoiceKind::Send));
    c.day += 99;
    assert!(lapse_petitions(&mut c).is_none(), "a SENT ask never lapses under them");
    resolve_sent_petition(&mut c, true);
    assert_eq!(c.gold, pay);
    assert!(standing_with(&c, &fid) > offset, "full credit");
    assert!(c.petition.is_none());

    // SEND, failed — a quarter, and no wage.
    let mut c = base.clone();
    c.gold = 0;
    stage_petition(&mut c, PetitionKind::Levy, 3);
    answer_petition(
        &mut c,
        PetitionChoiceKind::Send,
        PetitionCapacity { available_hands: 9, meals: 99 },
    )
    .unwrap();
    resolve_sent_petition(&mut c, false);
    assert_eq!(c.gold, 0);
    assert!((standing_with(&c, &fid) - (offset + w * 0.25)).abs() < 1e-9);
}

/// LAW: the lapse sweep runs BEFORE the storyteller, so today's expiry can
/// never be papered over by today's event.
#[test]
fn the_deadline_sweep_runs_before_the_storyteller() {
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty());
    c.director.points = 500; // the storyteller is ready to speak
    stage_petition(&mut c, PetitionKind::Relief, 3);
    // The ask falls due on the dawn we are about to fold.
    c.petition.as_mut().unwrap().expires_day = c.day + 1;
    let snap = snap_of(&c);
    let out = dawn_fold(&mut c, &snap, &[]);
    assert!(out.petition_lapsed.is_some(), "the deadline fell due in the fold");
    // Only a BRAND NEW ask may occupy the slot afterwards.
    assert!(c.petition.as_ref().is_none_or(|p| p.posted_day == c.day));
}

/// LAW: politics becomes personal through ONE line — a thought on the
/// colonists whose home ground was slighted, never a mind.
#[test]
fn home_ground_politics_reaches_the_colonists_as_a_thought() {
    let holder_of = |c: &Campaign, id: &str| -> Option<String> {
        c.founding
            .world
            .home_of(id)
            .and_then(|lm| faction_holding(&c.factions, lm))
            .filter(|f| crate::factions::faction_kind_spec(f.kind).petitions)
            .map(|f| f.id.clone())
    };
    let c = seed_where(|c| c.roster.iter().any(|id| holder_of(c, id).is_some()));
    let (member, fid) = c
        .roster
        .iter()
        .find_map(|id| holder_of(&c, id).map(|f| (id.clone(), f)))
        .expect("seed_where guaranteed one");

    let mut refuse = c.clone();
    stage_petition_for(&mut refuse, &fid, PetitionKind::Levy, 3);
    let thoughts =
        answer_petition(&mut refuse, PetitionChoiceKind::Refuse, PetitionCapacity::default())
            .unwrap();
    assert!(thoughts.contains(&(member.clone(), "home_refused".to_string())), "{thoughts:?}");

    let mut serve = c.clone();
    stage_petition_for(&mut serve, &fid, PetitionKind::Levy, 3);
    answer_petition(
        &mut serve,
        PetitionChoiceKind::Send,
        PetitionCapacity { available_hands: 9, meals: 99 },
    )
    .unwrap();
    let thoughts = resolve_sent_petition(&mut serve, true);
    assert!(thoughts.contains(&(member, "home_served".to_string())), "{thoughts:?}");
}

/// The storyteller's own petition arm: a hostile power DEMANDS (tribute), a
/// friendly one asks; the trope is faithfully ineligible with no powers.
#[test]
fn the_petition_trope_asks_and_hostile_powers_demand() {
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty());
    c.day = 30; // past the storyteller's 3-day cooldown
    c.director.points = 500;
    c.director.plan = Some(TropeKind::Petition);
    let view = DirectorView {
        wealth: 500,
        avg_mood: 60.0,
        petitioner_count: petitioners(&c.factions).len(),
        available_hands: 4,
        ..Default::default()
    };
    let tick = tick_director_full(&mut c, &view);
    assert!(tick.event.is_none(), "a petition carries no fixture payload");
    let p = tick.petition.expect("the ask opened");
    assert_ne!(p.kind, PetitionKind::Tribute, "a power that is not hostile ASKS");
    assert_eq!(p.expires_day, p.posted_day + PETITION_DAYS);
    assert!(p.need_hands >= 2, "always at least two hands (petitions.ts:298)");
    assert_eq!(p.need_provisions, p.need_hands * petition_spec(p.kind).days as u32);
    assert_eq!(c.petition.as_ref().map(|x| x.id.clone()), Some(p.id.clone()));
    assert!(c.chronicle.iter().any(|e| e.text.contains("asks for")));
    assert_eq!(c.faction_ledger.get(&p.faction_id).last_petition_day, c.day);

    // A power that has turned on you does not ask nicely.
    let mut c2 = seed_where(|c| !petitioners(&c.factions).is_empty());
    for f in petitioners(&c2.factions).iter().map(|f| f.id.clone()).collect::<Vec<_>>() {
        c2.faction_ledger.at(&f).hostile_since = Some(1);
    }
    c2.day = 30;
    c2.director.points = 500;
    c2.director.plan = Some(TropeKind::Petition);
    let tick = tick_director_full(&mut c2, &view);
    assert_eq!(tick.petition.expect("ask").kind, PetitionKind::Tribute);

    // No powers → faithfully ineligible (never committed, never fired).
    let mut plain = Campaign::new(new_founding(99, 0, ScenarioId::Village).expect("founding"));
    plain.director.points = 500;
    for _ in 0..150 {
        let snap = snap_of(&plain);
        let out = dawn_fold(&mut plain, &snap, &[]);
        assert!(out.petition_opened.is_none());
        assert!(out.event.as_ref().is_none_or(|e| e.kind != TropeKind::Petition));
        plain.raid = None;
    }
}

// ===========================================================================
// THE FOUNDERS' AMBITION
// ===========================================================================

#[test]
fn the_arc_rolls_from_the_country_that_actually_rolled() {
    let c = seed_where(|c| wild_power(&c.factions).is_some() && !petitioners(&c.factions).is_empty());
    let a = c.ambition.as_ref().expect("a country with powers has an arc");
    assert!((3..=5).contains(&a.stages.len()), "3-5 stages, got {}", a.stages.len());
    assert_eq!(a.stages[0].kind, StageKind::Company);
    assert_eq!(a.stages[0].target, (c.roster.len() as i64 + 2).max(5));
    assert_eq!(a.stages[1].kind, StageKind::Favour);
    assert_eq!(a.stages[1].target, 40);
    assert_eq!(a.stages[2].kind, StageKind::Prosper);
    assert_eq!(a.stages[2].target, 2600);
    assert_eq!(a.stages[3].kind, StageKind::Settle, "the wild power gets a stage");
    // The patron is a power that actually asks for things.
    let patron = a.stages[1].faction_id.clone().unwrap();
    assert!(petitioners(&c.factions).iter().any(|f| f.id == patron));
    assert!(a.achieved_day.is_none());

    // A country with no petitioning power rolls NO arc and takes NO draws.
    let mut rng = RngState::new(7);
    let before = rng.rng_counter;
    let none = roll_ambition(&mut rng, &[], &[], &c.founding.world);
    assert!(none.is_none());
    assert_eq!(rng.rng_counter, before, "the early return must draw nothing");
}

/// LAW: stages close IN ORDER, one per dawn, on a ZERO-DRAW sweep.
#[test]
fn stages_close_in_order_and_the_sweep_draws_nothing() {
    let mut c = seed_where(|c| wild_power(&c.factions).is_some());
    let patron = c.ambition.as_ref().unwrap().stages[1].faction_id.clone().unwrap();

    // Stage 3's condition is true, stage 1's is not — NOTHING closes.
    let rich = AmbitionTruth { roster_len: 1, wealth: 999_999, raids_won: 99 };
    let before = c.rng.rng_counter;
    assert!(check_ambition(&mut c, rich).is_none(), "a later stage cannot jump the queue");
    assert_eq!(c.rng.rng_counter, before, "checkAmbition must be zero-draw");

    // Now stand on your own feet: ONE stage closes, and only one.
    let target = c.ambition.as_ref().unwrap().stages[0].target as usize;
    let truth = AmbitionTruth { roster_len: target, wealth: 999_999, raids_won: 99 };
    let step = check_ambition(&mut c, truth).expect("stage one falls");
    match step {
        AmbitionStep::Stage { done, total, .. } => {
            assert_eq!(done, 1);
            assert!(total >= 4);
        }
        AmbitionStep::Achieved { .. } => panic!("the arc cannot finish on its first stage"),
    }
    assert!(c
        .chronicle
        .iter()
        .any(|e| e.headline.as_deref().is_some_and(|h| h.contains(" of "))));
    // Stage two waits on STANDING, which the wealth in `truth` cannot buy.
    assert!(check_ambition(&mut c, truth).is_none(), "one stage per dawn, in order");
    crate::petitions::move_standing(&mut c, &patron, 40.0);
    assert!(matches!(check_ambition(&mut c, truth), Some(AmbitionStep::Stage { done: 2, .. })));
}

/// LAW: the last stage ENDS THE CAMPAIGN, and the ending SPENDS the stories.
#[test]
fn the_last_stage_ends_the_campaign_with_an_epilogue() {
    let mut c = seed_where(|c| wild_power(&c.factions).is_some());
    let n = c.ambition.as_ref().unwrap().stages.len();
    // Make every political condition true at once.
    for f in c.factions.clone() {
        crate::petitions::move_standing(&mut c, &f.id, 100.0);
    }
    c.director.raids_won = 5;
    let truth = AmbitionTruth { roster_len: 50, wealth: 999_999, raids_won: 5 };
    let mut last = None;
    for i in 0..n {
        let step = check_ambition(&mut c, truth).unwrap_or_else(|| panic!("stage {i} must close"));
        last = Some(step);
    }
    assert!(matches!(last, Some(AmbitionStep::Achieved { .. })), "the arc completes");
    assert_eq!(c.ambition.as_ref().unwrap().achieved_day, Some(c.day));
    assert!(check_ambition(&mut c, truth).is_none(), "a finished arc closes nothing more");

    let ep = build_epilogue(&c, 999_999);
    assert_eq!(ep.lines.len(), c.roster.len(), "every companion still standing walks out");
    assert_eq!(ep.title, c.ambition.as_ref().unwrap().title);
    assert_eq!(ep.standings.len(), c.factions.len());
    // The ending is DATA drawn from the real record, not authored prose.
    assert!(ep.lines.iter().any(|l| l.band.is_some()));
    assert!(ep.lines.iter().filter(|l| l.was_heir).count() <= 1);
    for l in &ep.lines {
        assert!(!l.name.is_empty());
        if l.home_landmark.is_some() {
            assert!(l.home_holder.is_some(), "held ground names its holder");
        }
    }
}

/// And the dawn fold reports it, terminally.
#[test]
fn the_dawn_fold_reports_the_arc_and_the_ending() {
    let mut c = seed_where(|c| wild_power(&c.factions).is_some());
    for f in c.factions.clone() {
        crate::petitions::move_standing(&mut c, &f.id, 100.0);
    }
    c.director.raids_won = 5;
    c.gold = 1_000_000; // colony_wealth clears the prosper bar
    let stages = c.ambition.as_ref().unwrap().stages.len();
    let mut ended = false;
    for _ in 0..stages + 3 {
        // A roster big enough for the `company` stage, and standing kept high
        // against the lazy drift.
        let extra: Vec<String> = c
            .founding
            .cast
            .companions
            .iter()
            .map(|x| x.id.clone())
            .filter(|id| !c.roster.contains(id))
            .collect();
        c.roster.extend(extra);
        for f in c.factions.clone() {
            crate::petitions::move_standing(&mut c, &f.id, 100.0);
        }
        let snap = snap_of(&c);
        let out = dawn_fold(&mut c, &snap, &[]);
        if let Some(AmbitionStep::Achieved { .. }) = out.ambition {
            assert!(out.epilogue.is_some(), "the ending carries its epilogue");
            ended = true;
            break;
        }
    }
    assert!(ended, "the arc completes under the real fold");
}

// ===========================================================================
// BANDS
// ===========================================================================

fn a_non_founder_band(c: &Campaign) -> Option<String> {
    c.founding.cast.bands.iter().find(|b| !b.founders).map(|b| b.id.clone())
}

#[test]
fn patience_drains_only_while_unserved_and_notice_takes_two_days() {
    let mut c = seed_where(|c| a_non_founder_band(c).is_some());
    let band = a_non_founder_band(&c).unwrap();
    assert!(sign_band(&mut c, &band, SignHow::Coin));
    let start = patience_of(&c, &band);

    // SERVED: a raid fielded within 5 days holds the clock exactly still.
    c.director.last_raid_day = Some(c.day);
    tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
    assert_eq!(patience_of(&c, &band), start, "being used is service");

    // UNSERVED: 2 a day, +6 on a hungry day (goals.ts:250-252).
    c.director.last_raid_day = None;
    tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
    let after_quiet = patience_of(&c, &band);
    tick_band_goals(&mut c, BandTickInput { hungry_day: true, wealth: 0, beds: 0 });
    let after_hungry = patience_of(&c, &band);
    let quiet_drain = start - after_quiet;
    let hungry_drain = after_quiet - after_hungry;
    assert!(
        quiet_drain == 2 || quiet_drain == 3,
        "2/day (+1 for an unmet poach want), got {quiet_drain}"
    );
    assert_eq!(hungry_drain, quiet_drain + 6, "hunger wears on everyone");

    // Run it dry: NOTICE, then two days, then they ride out.
    let mut guard = 0;
    while status_of(&c, &band) == BandStatus::Signed {
        c.day += 1;
        tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
        guard += 1;
        assert!(guard < 200, "patience must run out");
    }
    assert_eq!(status_of(&c, &band), BandStatus::Notice);
    let members = band_members(&c, &band);
    assert!(members.iter().all(|m| c.roster.contains(m)), "notice is not departure");
    c.day += 1;
    tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
    assert!(members.iter().all(|m| c.roster.contains(m)), "one day is not two");
    c.day += 1;
    let report = tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
    assert!(!report.departed.is_empty(), "two days and they ride out");
    assert!(members.iter().all(|m| !c.roster.contains(m)));
    assert_eq!(status_of(&c, &band), BandStatus::Camped);
    assert_eq!(
        c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.state.times_departed,
        1
    );
}

/// LAW: the notice countdown FREEZES the moment they are served again.
#[test]
fn service_freezes_the_notice_countdown() {
    let mut c = seed_where(|c| a_non_founder_band(c).is_some());
    let band = a_non_founder_band(&c).unwrap();
    assert!(sign_band(&mut c, &band, SignHow::Coin));
    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band) {
        l.state.status = BandStatus::Notice;
        l.notice_day = Some(c.day);
    }
    let members = band_members(&c, &band);
    for _ in 0..20 {
        c.day += 1;
        c.director.last_raid_day = Some(c.day); // fielded, over and over
        tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: 0, beds: 0 });
    }
    assert!(members.iter().all(|m| c.roster.contains(m)), "served bands do not leave");
}

/// LAW: the founders never leave — exempt BY KIND, not by patience.
#[test]
fn the_founders_never_leave() {
    let mut c = seed_where(|c| !c.factions.is_empty());
    let founders = c.founding.cast.bands.iter().find(|b| b.founders).unwrap().id.clone();
    assert!(matches!(
        c.band_states.iter().find(|(id, _)| id == &founders).unwrap().1.state.goal,
        BandGoal::Guild
    ));
    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &founders) {
        l.state.patience = 0;
    }
    for _ in 0..60 {
        c.day += 1;
        tick_band_goals(&mut c, BandTickInput { hungry_day: true, wealth: 0, beds: 0 });
    }
    assert!(!c.roster.is_empty(), "the founders' exit is the guild-fall, not a departure");
    assert_eq!(status_of(&c, &founders), BandStatus::Signed);
}

/// LAW: patience clocks hold while afield — nobody rides out from the middle
/// of the road.
#[test]
fn a_band_with_people_afield_cannot_depart() {
    let mut c = seed_where(|c| a_non_founder_band(c).is_some());
    let band = a_non_founder_band(&c).unwrap();
    assert!(sign_band(&mut c, &band, SignHow::Coin));
    let members = band_members(&c, &band);

    let stacks = road_stacks();
    let ctx = DispatchContext { stacks: &stacks, unavailable: &[] };
    dispatch_party(
        &mut c,
        &members[..1],
        DispatchOpts { x: Some(20.0), z: Some(0.0), ..Default::default() },
        ctx,
    )
    .expect("dispatch");
    assert!(is_afield(&c, &members[0]));

    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band) {
        l.state.status = BandStatus::Notice;
        l.notice_day = Some(c.day - 5); // long overdue
    }
    tick_band_goals(&mut c, BandTickInput { hungry_day: true, wealth: 0, beds: 0 });
    assert!(
        members.iter().all(|m| c.roster.contains(m)),
        "the departure must hold while someone is on the road"
    );
    // And the notice clock was re-stamped to today, so it restarts on return.
    assert_eq!(
        c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.notice_day,
        Some(c.day)
    );
}

/// LAW: `cause_requested` is SETTABLE — the cause_raid trope was unreachable
/// until this landed (S7b flagged it).
#[test]
fn requesting_a_cause_unlocks_the_cause_raid_trope() {
    let mut c = seed_where(|c| {
        c.band_states
            .iter()
            .any(|(_, l)| matches!(l.state.goal, BandGoal::Deed { .. } | BandGoal::Debt { .. }))
    });
    let band = c
        .band_states
        .iter()
        .find(|(_, l)| matches!(l.state.goal, BandGoal::Deed { .. } | BandGoal::Debt { .. }))
        .map(|(id, _)| id.clone())
        .unwrap();
    assert!(request_cause(&mut c, &band), "a deed/debt band has a cause to take up");
    assert!(!request_cause(&mut c, &band), "once only");
    assert!(c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.cause_requested);

    // A prosperity/guild band has no cause to take up.
    if let Some(other) = c
        .band_states
        .iter()
        .find(|(_, l)| matches!(l.state.goal, BandGoal::Guild | BandGoal::Prosperity { .. }))
        .map(|(id, _)| id.clone())
    {
        assert!(!request_cause(&mut c, &other));
    }

    // And the storyteller can now reach it.
    c.director.points = 500;
    c.day = 10;
    let view = DirectorView { wealth: 400, avg_mood: 60.0, ..Default::default() };
    let tick = tick_director_full(&mut c, &view);
    let ev = tick.event.expect("the cause jumps the queue the moment it is affordable");
    assert_eq!(ev.kind, TropeKind::CauseRaid);
    assert_eq!(c.raid.as_ref().unwrap().band_ref.as_deref(), Some(band.as_str()));

    // Winning it SIGNS them and satisfies the goal.
    let raid = c.raid.clone().unwrap();
    let result = crate::campaign::RaidResultView {
        victory: true,
        gold_looted: 0,
        member_hp: c.roster.iter().map(|id| (id.clone(), 1.0)).collect(),
    };
    crate::campaign::resolve_raid(&mut c, &raid, &result, &InventorySnapshot::default());
    let live = c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.clone();
    assert_eq!(live.state.status, BandStatus::Signed);
    assert!(live.satisfied_day.is_some(), "their matter is settled");
    assert!(!live.cause_requested, "a resolved cause clears the request");
    assert_eq!(live.state.patience, 100, "a satisfied goal pins patience");
    // …and a pinned goal never drains again.
    for _ in 0..40 {
        c.day += 1;
        tick_band_goals(&mut c, BandTickInput { hungry_day: true, wealth: 0, beds: 0 });
    }
    assert_eq!(patience_of(&c, &band), 100);
}

/// LAW: coin POSTPONES, never deletes — and the price scales with proximity.
#[test]
fn sign_price_scales_with_proximity_desperation_and_history() {
    let mut c = seed_where(|c| {
        c.band_states.iter().any(|(_, l)| matches!(l.state.goal, BandGoal::Prosperity { .. }))
    });
    let band = c
        .band_states
        .iter()
        .find(|(_, l)| matches!(l.state.goal, BandGoal::Prosperity { .. }))
        .map(|(id, _)| id.clone())
        .unwrap();
    let discount = crate::scenario::scenario_spec(c.founding.scenario).sign_discount;
    let cost = crate::bands::band_cost(&c, &band) as f64;

    // Far from their goal: proximity is (0+0)/2 = 0 → the 0.5 floor.
    c.renown = 0;
    c.gold = 0;
    let far = sign_price(&c, &band);
    assert_eq!(far, crate::defs::js_round(discount * cost * 0.5));

    // MID-QUEST IS DEAR: at the threshold, proximity 1 → ×1.5.
    let (need_renown, need_gold) =
        match &c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.state.goal {
            BandGoal::Prosperity { need_renown, need_gold } => (*need_renown, *need_gold),
            _ => unreachable!(),
        };
    c.renown = need_renown;
    c.gold = need_gold;
    let near = sign_price(&c, &band);
    assert_eq!(near, crate::defs::js_round(discount * cost * 1.5));
    assert!(near > far, "proximity makes them dearer");

    // Desperation ×0.4 (the refugee_band window).
    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band) {
        l.desperate_until = Some(c.day + 3);
    }
    assert_eq!(sign_price(&c, &band), crate::defs::js_round(discount * cost * 1.5 * 0.4));
    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band) {
        l.desperate_until = None;
        l.state.times_departed = 2;
    }
    // A band that walked twice costs 1 + 0.25×2 = 1.5× more.
    assert_eq!(sign_price(&c, &band), crate::defs::js_round(discount * cost * 1.5 * 1.5));

    // A coin signing DEFERS the goal — it does not delete it.
    if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band) {
        l.state.times_departed = 0;
    }
    let goal_before =
        c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.state.goal.clone();
    assert!(sign_band(&mut c, &band, SignHow::Coin));
    let live = c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.clone();
    assert!(live.deferred_by_coin);
    assert!(live.satisfied_day.is_none(), "coin buys time, not settlement");
    assert_eq!(live.state.goal, goal_before);
    assert_eq!(live.state.patience, 40, "a coin signing starts at 40, not 60");
}

/// The prosperity bar is DERIVED from the same rolled numbers, never re-rolled
/// — and it wants a bed for every colonist.
#[test]
fn prosperity_signs_on_wealth_and_beds() {
    let mut c = seed_where(|c| {
        c.band_states.iter().any(|(_, l)| matches!(l.state.goal, BandGoal::Prosperity { .. }))
    });
    let band = c
        .band_states
        .iter()
        .find(|(_, l)| matches!(l.state.goal, BandGoal::Prosperity { .. }))
        .map(|(id, _)| id.clone())
        .unwrap();
    let goal = c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.state.goal.clone();
    let bar = need_wealth(&goal);
    let (need_renown, need_gold) = match &goal {
        BandGoal::Prosperity { need_renown, need_gold } => (*need_renown, *need_gold),
        _ => unreachable!(),
    };
    assert_eq!(bar, need_gold * 3 + need_renown * 5, "goals.ts needWealth");
    assert!(can_recruit(&c), "the village start has a labour market");

    // Wealth alone is not enough — a paymaster worth the name has beds.
    tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: bar, beds: 0 });
    assert_eq!(status_of(&c, &band), BandStatus::Camped);
    let beds = c.roster.len() + 10;
    let report =
        tick_band_goals(&mut c, BandTickInput { hungry_day: false, wealth: bar, beds });
    assert!(report.signed.contains(&band), "wealth AND beds signs them");
    assert_eq!(status_of(&c, &band), BandStatus::Signed);
    assert!(
        c.band_states.iter().find(|(id, _)| id == &band).unwrap().1.satisfied_day.is_some(),
        "their goal was the colony itself"
    );
    assert!(report.thoughts.iter().any(|(_, k)| k == "goal_served"));
}

/// A start with no labour market shuts every door at the SIM SEAM.
#[test]
fn a_closed_start_cannot_sign_anyone() {
    let mut c = political(0xC0_FFEE, ScenarioId::Town);
    assert!(!can_recruit(&c));
    let Some(band) = a_non_founder_band(&c) else { return };
    assert!(!sign_band(&mut c, &band, SignHow::Coin));
    assert_eq!(status_of(&c, &band), BandStatus::Camped);
}

// ===========================================================================
// AFIELD
// ===========================================================================

fn road_stacks() -> Vec<StackView> {
    vec![
        StackView { id: "s0".into(), item: "meal".into(), count: 40 },
        StackView { id: "s1".into(), item: "grain".into(), count: 40 },
    ]
}

fn no_fight(_: &Campaign, _: &crate::afield::AfieldParty) -> ErrandFight {
    ErrandFight { victory: true, gold: 0, member_hp: Vec::new() }
}

#[test]
fn dispatch_prices_the_road_and_packs_the_larder() {
    let mut c = seed_where(|c| c.roster.len() >= 2);
    let stacks = road_stacks();
    // 40 meals (1 each) + 40 grain (0.25 each) = 50 nutrition-days.
    assert!((larder_nutrition(&stacks) - 50.0).abs() < 1e-9);

    let riders: Vec<String> = c.roster.iter().take(2).cloned().collect();
    let opts = DispatchOpts { x: Some(60.0), z: Some(0.0), ..Default::default() };
    let ctx = DispatchContext { stacks: &stacks, unavailable: &[] };
    let cost = dispatch_cost(&c, &riders, &opts, ctx);
    assert_eq!(cost.travel_days, 3, "60 units at 20 a day");
    // n × (travelDays*2 + RATION_MARGIN) × RATION = 2 × 8 = 16.
    assert_eq!(cost.provisions, 16);
    assert!(cost.blocked.is_none());

    // MEALS FIRST — they travel best.
    let (packed, take) = pack_rations(&stacks, 16.0);
    assert_eq!(take, vec![("meal".to_string(), 16)]);
    assert!((packed - 16.0).abs() < 1e-9);
    // …then raw, when the meals run short.
    let thin = vec![
        StackView { id: "s0".into(), item: "meal".into(), count: 3 },
        StackView { id: "s1".into(), item: "grain".into(), count: 40 },
    ];
    let (packed, take) = pack_rations(&thin, 5.0);
    assert_eq!(take, vec![("meal".to_string(), 3), ("grain".to_string(), 8)]);
    assert!((packed - 5.0).abs() < 1e-9);

    // A thin larder BLOCKS, and says by how much.
    let bare = vec![StackView { id: "s0".into(), item: "meal".into(), count: 4 }];
    let bare_ctx = DispatchContext { stacks: &bare, unavailable: &[] };
    let cost = dispatch_cost(&c, &riders, &opts, bare_ctx);
    assert_eq!(cost.blocked.as_deref(), Some("the larder holds 4 of 16 days of rations"));
    assert!(dispatch_party(&mut c, &riders, opts.clone(), bare_ctx).is_none());

    // Nobody to send / someone who cannot ride.
    let cost = dispatch_cost(&c, &[], &opts, ctx);
    assert_eq!(cost.blocked.as_deref(), Some("nobody to send"));
    let laid_up = DispatchContext { stacks: &stacks, unavailable: &riders[..1] };
    let cost = dispatch_cost(&c, &riders, &opts, laid_up);
    assert_eq!(cost.blocked.as_deref(), Some("someone on that list cannot ride"));

    let d = dispatch_party(&mut c, &riders, opts, ctx).expect("dispatch");
    assert_eq!(d.take_items, vec![("meal".to_string(), 16)]);
    assert_eq!(d.away_ids, riders);
    assert!((d.party.provisions - 16.0).abs() < 1e-9);
}

/// LAW: away members are excluded from the colony's working strength — and
/// they cannot be dispatched twice.
#[test]
fn away_members_leave_the_colonys_strength() {
    let mut c = seed_where(|c| c.roster.len() >= 3);
    let stacks = road_stacks();
    let ctx = DispatchContext { stacks: &stacks, unavailable: &[] };
    let riders: Vec<String> = c.roster.iter().take(1).cloned().collect();
    let snap = snap_of(&c);
    let before = petition_capacity(&c, &snap).available_hands;
    dispatch_party(
        &mut c,
        &riders,
        DispatchOpts { x: Some(20.0), z: Some(0.0), ..Default::default() },
        ctx,
    )
    .expect("dispatch");
    let after = petition_capacity(&c, &snap).available_hands;
    assert_eq!(after, before - 1, "the road takes them off the answer's strength");
    assert!(is_afield(&c, &riders[0]));
    // No double-dispatch.
    assert!(dispatch_party(
        &mut c,
        &riders,
        DispatchOpts { x: Some(20.0), z: Some(0.0), ..Default::default() },
        ctx
    )
    .is_none());
    // A snapshot member the fixture reports as not ready is out too.
    let mut snap2 = snap_of(&c);
    if let Some(m) = snap2.members.get_mut(1) {
        m.ready = false;
    }
    assert_eq!(petition_capacity(&c, &snap2).available_hands, after - 1);
}

/// LAW: progress DERIVES from the day — a five-day jump lands the party in
/// exactly the same place as five one-day steps.
#[test]
fn afield_progress_is_jump_proof() {
    let mut stepper = seed_where(|c| c.roster.len() >= 2);
    let mut jumper = stepper.clone();
    let stacks = road_stacks();
    let ctx = DispatchContext { stacks: &stacks, unavailable: &[] };
    let riders: Vec<String> = stepper.roster.iter().take(2).cloned().collect();
    let opts = DispatchOpts {
        x: Some(40.0),
        z: Some(0.0),
        errand: Some(Errand::Scout),
        ..Default::default()
    };
    let a = dispatch_party(&mut stepper, &riders, opts.clone(), ctx).expect("dispatch");
    let b = dispatch_party(&mut jumper, &riders, opts, ctx).expect("dispatch");
    assert_eq!(a.party.travel_days, 2);
    assert_eq!(b.party.travel_days, 2);

    // The phases, straight off the calendar.
    assert_eq!(afield_phase(a.party.depart_day, &a.party), AfieldPhase::Outbound);
    assert_eq!(afield_phase(a.party.depart_day + 2, &a.party), AfieldPhase::Arrived);
    assert_eq!(afield_phase(a.party.depart_day + 3, &a.party), AfieldPhase::Homeward);
    assert_eq!(afield_phase(a.party.depart_day + 4, &a.party), AfieldPhase::Home);

    for _ in 0..5 {
        stepper.day += 1;
        sync_afield(&mut stepper, no_fight);
    }
    jumper.day += 5;
    let report = sync_afield(&mut jumper, no_fight);

    assert!(stepper.afield.is_empty() && jumper.afield.is_empty(), "both came home");
    assert_eq!(report.home_ids.len(), 2);
    assert_eq!(
        stepper.chronicle.iter().filter(|e| e.text.contains("came home")).count(),
        jumper.chronicle.iter().filter(|e| e.text.contains("came home")).count(),
        "the calendar, not the number of sweeps, decides where they are"
    );
}

/// LAW: run dry and they turn back hungry, errand undone.
#[test]
fn a_dry_company_turns_back() {
    let mut c = seed_where(|c| c.roster.len() >= 2);
    let stacks = road_stacks();
    let riders: Vec<String> = c.roster.iter().take(2).cloned().collect();
    let d = dispatch_party(
        &mut c,
        &riders,
        DispatchOpts {
            x: Some(120.0),
            z: Some(0.0),
            errand: Some(Errand::Scout),
            ..Default::default()
        },
        DispatchContext { stacks: &stacks, unavailable: &[] },
    )
    .expect("dispatch");
    assert_eq!(d.party.travel_days, 6);
    // Strip them to two days' food and watch the road do its work.
    c.afield[0].provisions = 4.0;

    let mut hungry_thoughts = 0;
    for _ in 0..14 {
        c.day += 1;
        let r = sync_afield(&mut c, no_fight);
        hungry_thoughts += r.thoughts.iter().filter(|(_, k)| k == "hungry_road").count();
        if c.afield.is_empty() {
            break;
        }
    }
    assert!(c.afield.is_empty(), "they came home");
    assert!(hungry_thoughts >= 4, "two members × two hungry days at least");
    assert!(c.chronicle.iter().any(|e| e.headline.as_deref() == Some("A company turns back")));
    assert!(c.chronicle.iter().any(|e| e.text.contains("came home empty-handed")));
}

/// The errands: an inquiry LEARNS a market, a scout FILES a report, and a
/// homecoming returns the leftovers and resolves the ask that sent them.
#[test]
fn errands_learn_report_and_answer() {
    // ---- INQUIRE ----
    let mut c = seed_where(|c| {
        c.founding.world.landmarks.iter().any(|l| market_kind_table(l.kind).is_some())
    });
    let market_lm = c
        .founding
        .world
        .landmarks
        .iter()
        .find(|l| market_kind_table(l.kind).is_some())
        .map(|l| l.id.clone())
        .unwrap();
    assert!(!market_known(&c, &market_lm), "knowledge is the resource — start ignorant");
    let stacks = road_stacks();
    let riders: Vec<String> = c.roster.iter().take(1).cloned().collect();
    let d = dispatch_party(
        &mut c,
        &riders,
        DispatchOpts {
            target_landmark_id: Some(market_lm.clone()),
            errand: Some(Errand::Inquire),
            ..Default::default()
        },
        DispatchContext { stacks: &stacks, unavailable: &[] },
    )
    .expect("dispatch");
    for _ in 0..(d.party.travel_days * 2 + 1) {
        c.day += 1;
        sync_afield(&mut c, no_fight);
    }
    assert!(market_known(&c, &market_lm), "the errand IS the learning");
    assert!(c.chronicle.iter().any(|e| e.headline.as_deref() == Some("Word of the market")));

    // ---- SCOUT (and the tracker perk makes the count EXACT) ----
    let mut c = seed_where(|c| c.roster.len() >= 2);
    c.threats.push(crate::director::ThreatState {
        id: "th1".to_string(),
        name: "Sarga's war-band".to_string(),
        comp: vec![("bandit".to_string(), 6)],
        tier: 1,
        target_landmark_id: "hall".to_string(),
        path: vec![(40.0, 0.0), (20.0, 0.0), (0.0, 0.0)],
        step_idx: 0,
        pillaged: Vec::new(),
    });
    assert!(!crate::knowledge::known_threat(&c, "th1"), "an unreported band is invisible");
    let riders: Vec<String> = c.roster.iter().take(1).cloned().collect();
    let d = dispatch_party(
        &mut c,
        &riders,
        DispatchOpts {
            x: Some(40.0),
            z: Some(0.0),
            errand: Some(Errand::Scout),
            scout_threat_ref: Some("th1".to_string()),
            tracker_aboard: true,
            ..Default::default()
        },
        DispatchContext { stacks: &stacks, unavailable: &[] },
    )
    .expect("dispatch");
    for _ in 0..(d.party.travel_days + 1) {
        c.day += 1;
        sync_afield(&mut c, no_fight);
    }
    let rep = c.threat_intel.iter().find(|(id, _)| id == "th1").expect("a report was filed");
    assert!(rep.1.exact, "a tracker's count is exact");
    assert_eq!(rep.1.power, 12, "6 bandits × power 2");
    assert_eq!(threat_strength_word(&c, "th1"), "12 spears, counted");

    // ---- THE PETITION the company was sent to answer ----
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty() && c.roster.len() >= 3);
    stage_petition(&mut c, PetitionKind::Levy, 2);
    let pid = c.petition.as_ref().unwrap().id.clone();
    let wage = petition_pay(c.petition.as_ref().unwrap());
    assert_eq!(wage, 8 * 2 * 3);
    let riders: Vec<String> = c.roster.iter().take(2).cloned().collect();
    let d = dispatch_party(
        &mut c,
        &riders,
        DispatchOpts {
            x: Some(20.0),
            z: Some(0.0),
            petition_id: Some(pid.clone()),
            comp: Some(vec![("looter".to_string(), 2)]),
            ..Default::default()
        },
        DispatchContext { stacks: &stacks, unavailable: &[] },
    )
    .expect("dispatch");
    assert_eq!(c.petition.as_ref().unwrap().chosen, Some(PetitionChoiceKind::Send));
    assert_eq!(c.petition.as_ref().unwrap().afield_id.as_deref(), Some(d.party.id.as_str()));
    c.gold = 0;
    let mut report = None;
    let mut all_hp: Vec<(String, f64)> = Vec::new();
    for _ in 0..(d.party.travel_days * 2 + 2) {
        c.day += 1;
        let r = sync_afield(&mut c, |_, p| ErrandFight {
            victory: true,
            gold: 25,
            member_hp: p.member_ids.iter().map(|id| (id.clone(), 0.3)).collect(),
        });
        all_hp.extend(r.set_hp.clone());
        if !r.arrivals.is_empty() {
            report = Some(r);
            break;
        }
    }
    let report = report.expect("they came home");
    // The wounds land the day the fight happens; the WARD is applied at the
    // gate (the TS split between `resolveErrand` and `comeHome`).
    assert!(all_hp.iter().any(|(_, f)| (*f - 0.3).abs() < 1e-9));
    assert!(c.petition.is_none(), "the ask closed when they got home");
    assert_eq!(c.gold, 25 + wage, "loot AND the wage came home");
    assert!(report.rations_returned > 0, "leftovers go back on the pile by the door");
    assert!(report.injuries.iter().any(|(_, d)| *d == 2), "hp under 0.4 beds them for two days");
    assert!(report.thoughts.iter().any(|(_, k)| k == "victory"));
}

/// The dawn fold's step 2 is live when the caller brings a resolver.
#[test]
fn the_dawn_fold_advances_the_road() {
    let mut c = seed_where(|c| c.roster.len() >= 2);
    let stacks = road_stacks();
    let riders: Vec<String> = c.roster.iter().take(1).cloned().collect();
    dispatch_party(
        &mut c,
        &riders,
        DispatchOpts {
            x: Some(20.0),
            z: Some(0.0),
            errand: Some(Errand::Scout),
            ..Default::default()
        },
        DispatchContext { stacks: &stacks, unavailable: &[] },
    )
    .expect("dispatch");
    let mut fight = no_fight;
    let mut home = false;
    for _ in 0..6 {
        let snap = snap_of(&c);
        let (_, road) = dawn_fold_political(&mut c, &snap, &[], &mut fight);
        if !road.arrivals.is_empty() {
            assert_eq!(road.arrivals[0].1, AfieldOutcome::Done);
            home = true;
            break;
        }
        assert_eq!(road.away_ids, riders, "away while the road lasts");
    }
    assert!(home, "the fold walked them home");
}

// ===========================================================================
// MARKETS + KNOWLEDGE
// ===========================================================================

/// LAW: prices DERIVE — zero draws, the same market every founding.
#[test]
fn market_prices_derive_and_draw_nothing() {
    let mut c = seed_where(|c| {
        c.founding.world.landmarks.iter().any(|l| market_kind_table(l.kind).is_some())
    });
    let lm = c
        .founding
        .world
        .landmarks
        .iter()
        .find(|l| market_kind_table(l.kind).is_some())
        .map(|l| l.id.clone())
        .unwrap();
    let before = c.rng.rng_counter;
    let a = market_at(&c, &lm, market_epoch(c.day)).expect("a settled place keeps a market");
    let b = market_at(&c, &lm, market_epoch(c.day)).unwrap();
    assert_eq!(a, b, "the same market every read");
    assert_eq!(c.rng.rng_counter, before, "market pricing must draw NOTHING");
    for (_, price) in &a {
        assert!(*price >= 1, "a price is never free");
    }
    // The wild kinds keep no market at all.
    for l in &c.founding.world.landmarks {
        if market_kind_table(l.kind).is_none() {
            assert!(market_at(&c, &l.id, 0).is_none(), "{:?} should keep no market", l.kind);
        }
    }
    // A sacked place keeps no market either.
    c.sacked.push(lm.clone());
    assert!(market_at(&c, &lm, market_epoch(c.day)).is_none());
    c.sacked.clear();

    // KNOWLEDGE AGES: the season turns and the jitter re-rolls.
    let mut moved = false;
    for e in 1..12 {
        if market_at(&c, &lm, e) != Some(a.clone()) {
            moved = true;
            break;
        }
    }
    assert!(moved, "prices must drift across seasons");
    assert_eq!(market_epoch(0), 0);
    assert_eq!(market_epoch(MARKET_SEASON_DAYS), 1);
    assert_eq!(market_epoch(MARKET_SEASON_DAYS * 3 - 1), 2);
}

/// LAW: only a KNOWN market can provision, and distance becomes a number.
#[test]
fn only_a_known_market_provisions_and_haulage_prices_the_road() {
    let far_enough = |c: &Campaign| {
        c.founding
            .world
            .landmarks
            .iter()
            .any(|l| market_kind_table(l.kind).is_some() && (l.x * l.x + l.z * l.z).sqrt() > 60.0)
    };
    let mut c = seed_where(far_enough);
    let far = c
        .founding
        .world
        .landmarks
        .iter()
        .find(|l| market_kind_table(l.kind).is_some() && (l.x * l.x + l.z * l.z).sqrt() > 60.0)
        .map(|l| l.id.clone())
        .unwrap();

    c.provision_source = Some(far.clone());
    assert!(provision_choice(&c).is_none(), "an unknown market cannot be a source");
    assert!(believed_market(&c, &far).is_none());

    c.market_intel.push((far.clone(), c.day));
    let choice = provision_choice(&c).expect("a known market provisions");
    assert_eq!(choice.landmark_id, far);
    let haul = haulage_from(&c, &far);
    assert!(haul >= 1, "a far market carries a surcharge (+1g per three road-days)");
    let market = market_at(&c, &far, market_epoch(c.day)).unwrap();
    let raw = market.iter().find(|(i, _)| i == &choice.item).unwrap().1;
    assert_eq!(choice.unit, raw + haul, "haulage rides on the price");
    // It picks the best gold PER NUTRITION, not the cheapest sticker.
    let mine = choice.unit as f64 / crate::defs::item_nutrition(&choice.item);
    for (item, price) in &market {
        let n = crate::defs::item_nutrition(item);
        if n <= 0.0 {
            continue;
        }
        assert!((price + haul) as f64 / n >= mine - 1e-9, "{item} was the better buy");
    }
    // The belief carries its age, and states when the season has turned.
    let believed = believed_market(&c, &far).expect("seen");
    assert!(!believed.stale && believed.age == 0);
    c.day += MARKET_SEASON_DAYS + 1;
    let believed = believed_market(&c, &far).expect("still remembered");
    assert!(believed.stale, "the season has turned since anyone looked");
    assert!(believed.text.contains("days back"));

    // The nearest market is a real derivation.
    let nearest = nearest_market(&c, None).expect("some settlement");
    assert!(market_kind_table(c.founding.world.landmark_by_id(&nearest).unwrap().kind).is_some());
}

/// LAW: staleness IS uncertainty — a rough count prints as a range that WIDENS
/// with the report's age (`knowledge.ts:59`).
#[test]
fn a_rough_report_widens_with_age() {
    let mut c = seed_where(|_| true);
    c.day = 10;
    let comp = vec![("bandit".to_string(), 10)]; // power 20
    c.threats.push(crate::director::ThreatState {
        id: "th1".to_string(),
        name: "Mazok's war-band".to_string(),
        comp: comp.clone(),
        tier: 1,
        target_landmark_id: "hall".to_string(),
        path: vec![(80.0, 0.0), (0.0, 0.0)],
        step_idx: 0,
        pillaged: Vec::new(),
    });
    crate::director::file_threat_report(&mut c, "th1", &comp, (80.0, 0.0), false);
    let widths: Vec<i64> = (0..6)
        .map(|d| {
            c.day = 10 + d;
            let w = threat_strength_word(&c, "th1");
            let nums: Vec<i64> = w
                .split(|ch: char| !ch.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(nums.len(), 2, "a rough count prints as a RANGE: {w}");
            nums[1] - nums[0]
        })
        .collect();
    for pair in widths.windows(2) {
        assert!(pair[1] >= pair[0], "the range must widen: {widths:?}");
    }
    assert!(widths[5] > widths[0], "five days of staleness is visibly wider");
    // An unreported band is INVISIBLE, not fuzzy.
    assert_eq!(threat_strength_word(&c, "nope"), "strength unknown");
    assert!(crate::knowledge::describe_threat_report(&c, "nope").contains("nobody has seen"));
    assert!(crate::knowledge::describe_threat_report(&c, "th1").contains("Mazok"));
}

// ===========================================================================
// THE FOLD, END TO END
// ===========================================================================

/// A politics campaign survives a long fold, keeps its invariants, and
/// round-trips through the save.
#[test]
fn a_political_campaign_folds_and_saves() {
    let mut c = seed_where(|c| !petitioners(&c.factions).is_empty());
    let mut asks = 0;
    let mut lapses = 0;
    for _ in 0..150 {
        let snap = ColonySnapshot {
            inventory: InventorySnapshot {
                stacks: vec![StackView { id: "s0".into(), item: "meal".into(), count: 30 }],
                buildings: vec![BuildingView {
                    id: "b0".into(),
                    kind: "bed".into(),
                    q: 0,
                    r: 0,
                    built: true,
                }],
            },
            sown_cells: Vec::new(),
            members: c
                .roster
                .iter()
                .map(|id| MemberView { id: id.clone(), mood: 55.0, starving_days: 0, ready: true })
                .collect(),
            rotted_food_units: 0.0,
            starved_today: Vec::new(),
        };
        let out = dawn_fold(&mut c, &snap, &[]);
        if out.petition_opened.is_some() {
            asks += 1;
        }
        if out.petition_lapsed.is_some() {
            lapses += 1;
        }
        if let Some(r) = &c.raid.clone() {
            // Resolve it so the storyteller keeps speaking.
            let result = crate::campaign::RaidResultView {
                victory: c.day % 2 == 0,
                gold_looted: 10,
                member_hp: c.roster.iter().map(|id| (id.clone(), 0.9)).collect(),
            };
            crate::campaign::resolve_raid(&mut c, r, &result, &snap.inventory);
        }
    }
    assert!(asks > 0, "the petition trope fires when powers exist");
    assert!(lapses > 0, "unanswered asks fall due — silence has a price");
    // Standing never leaves its band.
    for f in &c.factions {
        let v = standing_with(&c, &f.id);
        assert!((-100.0..=100.0).contains(&v), "{} standing {v}", f.name);
    }
    let json = serde_json::to_string(&c).expect("serialize");
    let back: Campaign = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, c, "the politics layer round-trips exactly");
    assert!(json.contains("politics_enabled"));
}
