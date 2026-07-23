//! `webband_play` — THE WEBBAND PORT, PLAYABLE.
//!
//! S9's honest gap read: *"NO RAID CAN BE STAGED UNDER `play`: raid activation
//! is a test-side injection seam, so the campaign loop is reachable only from
//! these integration tests, never from a shipped binary."* S12 closed it. It
//! opens the same window `engine_play`'s `play` opens, over the same
//! `webband_colony` fixture, but the thing driving the clock is the CAMPAIGN:
//!
//!   seeded founding (`webband_app::founding`) -> roster seated on colonist
//!   bodies -> the colony works in real time, visible -> at every 600th tick
//!   the dawn fold runs (provisioner / exodus / trade / THE STORYTELLER with
//!   its committed-plan draw) -> the trope it commits is written back through
//!   the fixture's own seams -> a raid MUSTERS AT THE RIM AND IS FOUGHT ON
//!   SCREEN -> `resolve_raid` folds the outcome -> the chronicle accumulates.
//!
//! **S13 CLOSED THE TWO GAPS S12 LEFT.**
//!
//! 1. **THE GUILD LAYER IS LIVE.** S11 ported factions / petitions / the
//!    founders' arc / band clocks / the road behind
//!    `Campaign::new_political`, and NOTHING turned it on — in the shipped
//!    binary those systems did not exist. They are now the DEFAULT
//!    (`--no-politics` opts out), so the game a player gets is the whole game:
//!    powers ask things of the guild, an ask has a deadline, silence costs
//!    more than a refusal, standing drifts asymmetrically, and the founders'
//!    ambition can END the campaign. WHY DEFAULT-ON: a mode nobody selects is
//!    a mode nobody plays, and S12's evidence for that is this exact bug. The
//!    apolitical path is kept, unchanged and reachable, for ONE reason — it is
//!    what the recorded soak/spine digests were taken against, and
//!    `--no-politics --headless 16 --digest` still reproduces the spine's
//!    three hashes bit for bit.
//! 2. **THE PLAYER HAS HANDS.** S12 bound pause/speed/save/chronicle and no
//!    ORDERS. Webband removed turns precisely because the orders grammar
//!    carries agency in real time; without it this was a watchable colony, not
//!    a game. `[Tab]` selects a colonist and `G/H/F/Y/X` write their directive
//!    fields — the same fields `raid_directives` pins (hold anchors at 3.95 u
//!    against a 9.54 u undirected chase; focus retargets the line) — plus
//!    `[V]` for the colony-side standing order (the per-colonist priority
//!    table, Webband's inspector Task row) and `7/8/9` for the petition's
//!    three answers, taken through `petition_choices` so a blocked answer
//!    prints the sim's own reason and never happens.
//!
//! **The loop is not re-implemented here.** It is
//! [`webband_bridge::Bridge`] — the exact code `sims --test webband_campaign`
//! (the 60-day soak) and `--test webband_spine` pin.
//!
//! **DETERMINISM IS UNAFFECTED BY THE PLAYER'S CLOCK.** Pause, speed and frame
//! rate change only how many `step_one`s land per frame; the fold fires on the
//! tick index, never on wall-clock time. ORDERS, of course, change the world —
//! that is what they are for — which is why the political soak drives its
//! answers from a `--petition-answer` POLICY applied at dawn rather than from
//! a keypress: a policy is reproducible, a human's timing is not.
//!
//! USAGE
//! ```text
//! webband_play                                   # windowed, village, politics LIVE
//! webband_play --no-politics                     # the pre-S13 campaign (the digest path)
//! webband_play --speed 3                         # start at x16
//! webband_play --force-raid-day 2                # ask the guild's enemies for one, early
//! webband_play --exit-after-secs 45              # bounded run (never leave a window up)
//! webband_play --headless 60 --petition-answer refuse --digest d.txt
//! ```
//!
//! CONTROLS (the fixture declares no `controls {}` block; these are the
//! HOST's, served through `controls_descriptor()`):
//!   `space` pause/resume · `1`..`4` speed x1/x4/x16/x64 · `R` force a raid ·
//!   `S` save · `C` chronicle · `0` the guild report ·
//!   **`Tab`/`N` select the next colonist · `G` guard · `H` hold · `F` focus ·
//!   `Y` harry · `X` clear the order · `V` set their trade ·
//!   `7` send / `8` pay / `9` refuse the open ask.**

use std::collections::HashMap;
use std::path::PathBuf;

use engine_play::player::{Player, PlayerConfig};
use engine_play_api::PlayableRuntime;
use engine_ui::{UiModel, Widget};

use webband_app::ambition::{ambition_progress, current_stage, describe_stage};
use webband_app::campaign::ColonySnapshot;
use webband_app::defs::{colony_wealth, food_days};
use webband_app::raids::{spawn_raid, SpawnOpts};
use webband_bridge::{
    describe_petition, fnv1a, petitioner_name, petitioners, sig_hash, standing_with, standing_word,
    Bridge, PetitionChoiceKind, ScenarioId, AGENTS, CAMPAIGN_SEED, DAY_TICKS, SEED,
};

/// Fixture steps per rendered frame, by speed index. x1 is "watch a colonist
/// walk"; x64 is "get me to the next raid". A colony day is 600 ticks, so at
/// ~28 frames/s in a debug build these are roughly 21 s / 5 s / 1.3 s / 0.3 s
/// per campaign day.
const SPEEDS: [u32; 4] = [1, 4, 16, 64];

/// The host's own key bindings, in the frozen `ControlsDescriptor` shape.
/// `Press` mode = one write per key-down edge, which is what a toggle needs.
const HOST_CONTROLS: &str = r#"{"bindings":[
  {"key":"space","field":"host.pause","value":1.0,"mode":"Press"},
  {"key":"1","field":"host.speed","value":0.0,"mode":"Press"},
  {"key":"2","field":"host.speed","value":1.0,"mode":"Press"},
  {"key":"3","field":"host.speed","value":2.0,"mode":"Press"},
  {"key":"4","field":"host.speed","value":3.0,"mode":"Press"},
  {"key":"r","field":"host.force_raid","value":1.0,"mode":"Press"},
  {"key":"s","field":"host.save","value":1.0,"mode":"Press"},
  {"key":"c","field":"host.chronicle","value":1.0,"mode":"Press"},
  {"key":"tab","field":"host.select_next","value":1.0,"mode":"Press"},
  {"key":"n","field":"host.select_next","value":1.0,"mode":"Press"},
  {"key":"g","field":"host.order_guard","value":1.0,"mode":"Press"},
  {"key":"h","field":"host.order_hold","value":1.0,"mode":"Press"},
  {"key":"f","field":"host.order_focus","value":1.0,"mode":"Press"},
  {"key":"y","field":"host.order_harry","value":1.0,"mode":"Press"},
  {"key":"x","field":"host.order_clear","value":1.0,"mode":"Press"},
  {"key":"v","field":"host.order_trade","value":1.0,"mode":"Press"},
  {"key":"7","field":"host.answer_send","value":1.0,"mode":"Press"},
  {"key":"8","field":"host.answer_pay","value":1.0,"mode":"Press"},
  {"key":"9","field":"host.answer_refuse","value":1.0,"mode":"Press"},
  {"key":"0","field":"host.report","value":1.0,"mode":"Press"}
]}"#;

/// The eight work columns of the colony's priority table, in the fixture's own
/// field order. `[V]` cycles the selected colonist through them (and past the
/// end, back to "no preference") — Webband's inspector Task row, reduced to a
/// keyboard.
const JOBS: [&str; 8] = ["chop", "forage", "hunt", "build", "cook", "craft", "haul", "grow"];
/// Points added to the chosen column. The fixture scores a job as
/// `band + pri * pri_scale(200) - distance`, and the widest gap between two
/// ordinary work bands is `band_haul 4000 - band_tend 2500 = 1500`, so 8
/// points (1600) lifts ANY chosen trade above every other work band while
/// staying far below the needs bands (eat 6000 / nap 5500) — i.e. exactly
/// Webband's "priority 1": their first choice of WORK, never a reason to
/// starve.
const PRI_POINTS: f32 = 8.0;

/// The mana sentinel the SELECTED colonist's snapshot view carries so the
/// renderer paints them differently. Presentation only: it is written into a
/// COPY of the frame's agent views, never into the sim (this binary's whole
/// discipline — nothing in the render path writes sim state).
const SEL_MANA: f32 = 7.0;

// ---------------------------------------------------------------------------
// The HUD cache.

/// Everything the HUD prints, refreshed on a bounded cadence so the window
/// never pays a GPU readback per widget per frame. Daily values come free
/// from the dawn fold's own snapshot; live values (raiders standing, downed
/// colonists) are refreshed every [`HUD_PERIOD`] ticks and at every dawn.
#[derive(Default, Clone)]
struct Hud {
    day: f32,
    day_min: f32,
    hour: f32,
    hands: f32,
    gold: f32,
    renown: f32,
    food_units: f32,
    food_days: f32,
    wealth: f32,
    raiders: f32,
    downed: f32,
    raids: f32,
    won: f32,
    lost: f32,
    staged: f32,
    speed: f32,
    paused: f32,
    away: f32,
    /// S13 — the TEXT channel (`engine_play::PlayerConfig::hud_texts`): the
    /// things a simulation carries that are not numbers. S12's honest limit
    /// ("the HUD is numeric — no time-of-day WORD, no event TITLE") was a
    /// missing `UiData` feature, not a fact about campaigns.
    texts: HashMap<String, String>,
}

const HUD_PERIOD: u32 = 30;

impl Hud {
    fn get(&self, key: &str) -> Option<f32> {
        Some(match key {
            "wb_day" => self.day,
            "wb_daymin" => self.day_min,
            "wb_hour" => self.hour,
            "wb_hands" => self.hands,
            "wb_gold" => self.gold,
            "wb_renown" => self.renown,
            "wb_food" => self.food_units,
            "wb_fooddays" => self.food_days,
            "wb_wealth" => self.wealth,
            "wb_raiders" => self.raiders,
            "wb_downed" => self.downed,
            "wb_raids" => self.raids,
            "wb_won" => self.won,
            "wb_lost" => self.lost,
            "wb_staged" => self.staged,
            "wb_speed" => self.speed,
            "wb_paused" => self.paused,
            "wb_away" => self.away,
            _ => return None,
        })
    }

    fn keys() -> Vec<String> {
        [
            "wb_day",
            "wb_daymin",
            "wb_hour",
            "wb_hands",
            "wb_gold",
            "wb_renown",
            "wb_food",
            "wb_fooddays",
            "wb_wealth",
            "wb_raiders",
            "wb_downed",
            "wb_raids",
            "wb_won",
            "wb_lost",
            "wb_staged",
            "wb_speed",
            "wb_paused",
            "wb_away",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    fn text_keys() -> Vec<String> {
        ["wb_sel", "wb_order", "wb_ask", "wb_powers", "wb_arc", "wb_note"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The campaign as a PlayableRuntime.

/// The adapter that makes a whole CAMPAIGN look like one `PlayableRuntime` to
/// the generic player: `step()` spends this frame's ticks on the fixture and
/// runs the dawn fold when the day rolls over, `agent_snapshot()` is the
/// fixture's own (so the renderer paints the real colony, raiders included),
/// `view_value()`/`view_text()` answer the HUD's named scalars and prose, and
/// `set_input()` receives the host key bindings the player resolved.
///
/// It owns the [`Bridge`], which owns the `GeneratedRuntime` — that ownership
/// chain is the whole trick: the generic player never needs to know a
/// campaign exists, and the bridge never needs to know a window exists.
struct CampaignRuntime {
    b: Bridge,
    paused: bool,
    speed_idx: usize,
    hud: Hud,
    hud_at: u32,
    /// Campaign log lines already echoed to the console.
    logged: usize,
    /// Wall-clock deadline after which the process closes ITSELF (`None` =
    /// never). A bounded harness must never be able to leave a window
    /// running, and the harness's own kill is the second bound, not the only
    /// one. Wall clock, deliberately: it is the only thing that bounds a run
    /// regardless of speed setting or how fast the GPU turns out to be — and
    /// it can do that without touching determinism, because it only ever
    /// STOPS the process, never changes how many ticks a frame spends.
    deadline: Option<std::time::Instant>,
    /// `--force-raid-day D`: ask for a raid once, at the first dawn of day D.
    force_raid_day: Option<i64>,
    /// `--save-dir`: where `[S]` (and a finished `--headless` run) writes
    /// BOTH halves of a save — `campaign.json` (host) + `fixture.bin` (the
    /// GPU state image). See `webband_bridge::persist`.
    save_dir: PathBuf,
    saves: u32,

    // -- S13: the player's hands -------------------------------------------
    /// Per-slot `stagger` values, read once at founding. A directive names its
    /// ward/target by STAGGER (the fixture's stable cross-agent id), never by
    /// slot — see the fixture header.
    stagger: Vec<u32>,
    /// Index into `b.slot_map` of the SELECTED colonist (clamped on every use;
    /// the roster shrinks when people walk out).
    sel: usize,
    /// Cycling cursor for the guard WARD (an ally) and the focus TARGET (a
    /// raider of the staged cohort).
    ward_cur: usize,
    foe_cur: usize,
    /// Cycling cursor into [`JOBS`] for `[V]`; `JOBS.len()` = no preference.
    trade_cur: usize,
    /// The last thing the player did, in the sim's own words — printed on the
    /// HUD so a blocked order explains itself where the player is looking.
    note: String,
    /// A dawn POLICY that answers the open petition (headless soaks — a
    /// keypress is not reproducible, a policy is).
    auto_answer: Option<PetitionChoiceKind>,
    /// The render descriptor with the selection band prepended (leaked once).
    render_desc: &'static str,
}

impl CampaignRuntime {
    fn new(
        mut b: Bridge,
        speed_idx: usize,
        exit_after_secs: f32,
        force_raid_day: Option<i64>,
        auto_answer: Option<PetitionChoiceKind>,
    ) -> Self {
        let render_desc = selection_render_descriptor(b.state.render_descriptor());
        // Staggers are seeded at spawn and never change, so ONE readback at
        // founding serves every order the player will ever give.
        let stagger = {
            let n = b.w.n;
            let buf = b.state.agent_stagger_buf.clone();
            webband_bridge::read_u32(&mut b.state, &buf, n)
        };
        let mut rt = CampaignRuntime {
            b,
            paused: false,
            speed_idx,
            hud: Hud::default(),
            hud_at: u32::MAX,
            logged: 0,
            deadline: if exit_after_secs > 0.0 {
                Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(exit_after_secs))
            } else {
                None
            },
            force_raid_day,
            save_dir: PathBuf::from("target/webband_play/save"),
            saves: 0,
            stagger,
            sel: 0,
            ward_cur: 0,
            foe_cur: 0,
            trade_cur: JOBS.len(),
            note: "no orders given".to_string(),
            auto_answer,
            render_desc,
        };
        rt.refresh_hud();
        rt
    }

    /// The whole day-boundary fold, plus the console echo of whatever the
    /// campaign just decided. Called from `step()` the instant the tick count
    /// crosses a multiple of 600 — never on a timer.
    fn dawn(&mut self) {
        let out = self.b.dawn();
        if let Some(d) = self.force_raid_day {
            if self.b.campaign.day >= d {
                self.force_raid_day = None;
                self.force_raid();
            }
        }
        // THE ANSWER POLICY (headless): take the standing answer the moment an
        // ask is open, at the dawn it opened, so the run is reproducible.
        if let Some(choice) = self.auto_answer {
            if self.b.campaign.petition.as_ref().is_some_and(|p| p.chosen.is_none()) {
                match self.b.answer(choice) {
                    Ok(line) => println!("[politics] day {} {line}", self.b.campaign.day),
                    Err(why) => println!(
                        "[politics] day {} could not {choice:?}: {why}",
                        self.b.campaign.day
                    ),
                }
            }
        }
        if let Some(t) = &self.b.achieved.clone() {
            println!(
                "[webband_play] THE FOUNDERS' AMBITION IS ACHIEVED on day {} — \"{t}\". \
                 The story is over, and won.",
                self.b.campaign.day
            );
            self.print_epilogue(&out);
        }
        if out.fell {
            println!(
                "[webband_play] THE GUILD FELL on day {} — the roster is empty.",
                self.b.campaign.day
            );
        }
        self.refresh_hud();
        self.drain_log();
    }

    fn print_epilogue(&self, out: &webband_bridge::DawnOutcome) {
        let Some(e) = &out.epilogue else { return };
        println!(
            "[epilogue] \"{}\" — day {}, renown {}, gold {}, wealth {}",
            e.title, e.day, e.renown, e.gold, e.wealth
        );
        for (name, st) in &e.standings {
            println!("[epilogue]   {name}: {st:.1} ({})", standing_word(*st));
        }
        for l in &e.lines {
            println!(
                "[epilogue]   {} of {} — home {} held by {} ({:.1}){}",
                l.name,
                l.band.clone().unwrap_or_else(|| "no band".into()),
                l.home_landmark.clone().unwrap_or_else(|| "nowhere".into()),
                l.home_holder.clone().unwrap_or_else(|| "no one".into()),
                l.home_standing,
                if l.was_heir { " — the arc named them" } else { "" }
            );
        }
    }

    /// Echo the bridge's own campaign log (the same lines the soak hashes).
    fn drain_log(&mut self) {
        while self.logged < self.b.log.len() {
            println!("[campaign] {}", self.b.log[self.logged]);
            self.logged += 1;
        }
    }

    /// THE DEBUG TROPE (`R`, or `--force-raid-day D`). It does NOT fake a
    /// raid: it rolls a real `spawn_raid` off the campaign's own seeded
    /// stream with the day/wealth/roster the colony actually has and parks it
    /// in `campaign.raid`, exactly where the storyteller's `Raid` trope parks
    /// its own. The next dawn's fold picks it up through `raid_tomorrow` and
    /// stages it through the ordinary seam, so everything downstream — the
    /// muster schedule, the fixture's warning, the combat, `resolve_raid` —
    /// is the shipped path. What it skips is only the STORYTELLER'S DECISION
    /// (accrual, plan, mercy gate).
    ///
    /// It draws from the campaign rng, so a forced campaign diverges from an
    /// unforced one from that point. That is honest and stated: this is a
    /// demo/debug verb, not a player verb.
    fn force_raid(&mut self) {
        if self.b.staged.is_some() {
            println!("[webband_play] force-raid ignored: a raid is already staged.");
            return;
        }
        if self.b.campaign.raid.is_some() {
            println!("[webband_play] force-raid ignored: a raid is already inbound.");
            return;
        }
        let inv = self
            .b
            .last_snap
            .as_ref()
            .map(|s: &ColonySnapshot| s.inventory.clone())
            .unwrap_or_default();
        let c = &mut self.b.campaign;
        let wealth = colony_wealth(c.gold, c.roster.len(), &inv);
        // day + 1: the fold that stages it will have incremented the day, and
        // step 24 keeps only a raid whose `arrives_day == day + 1`.
        let raid = spawn_raid(
            &mut c.rng,
            c.day + 1,
            wealth,
            c.roster.len(),
            c.founding.scenario,
            SpawnOpts::default(),
        );
        println!(
            "[webband_play] FORCED RAID: {} bodies, tier {}, elite {:?} — they will be at the \
             fences the morning after next.",
            raid.comp.iter().map(|(_, n)| i64::from(*n)).sum::<i64>(),
            raid.tier,
            raid.elite_name
        );
        c.raid = Some(raid);
    }

    // -- S13: selection + orders -------------------------------------------

    /// The selected colonist as (roster id, body slot), clamped to the live
    /// roster. `None` only when the colony is empty.
    fn selected(&mut self) -> Option<(String, usize)> {
        if self.b.slot_map.is_empty() {
            return None;
        }
        self.sel %= self.b.slot_map.len();
        Some(self.b.slot_map[self.sel].clone())
    }

    fn name_of(&self, id: &str) -> String {
        self.b.campaign.companion_display_name(id)
    }

    fn select_next(&mut self) {
        if self.b.slot_map.is_empty() {
            return;
        }
        self.sel = (self.sel + 1) % self.b.slot_map.len();
        let Some((id, _)) = self.selected() else { return };
        let name = self.name_of(&id);
        self.note = format!("{name} selected");
        println!("[orders] selected {name}");
        self.refresh_hud();
    }

    /// Write the three directive fields for the selected colonist. THIS IS THE
    /// WHOLE ORDERS PATH: the fixture's combat verbs read `directive_kind` /
    /// `directive_target` / `directive_pos` in their masks, scores and
    /// steering, and `raid_directives` (S5, re-measured on the ability path at
    /// S5c) pins that they measurably change what a colonist does. Nothing
    /// here is a new mechanism — the player is writing the fields the test
    /// writes.
    fn set_directive(&mut self, kind: u32, target: u32, pos: Option<[f32; 3]>) {
        let Some((_, slot)) = self.selected() else { return };
        let s = &self.b.state;
        webband_bridge::write_u32(s, &s.agent_directive_kind_buf, slot, kind);
        webband_bridge::write_u32(s, &s.agent_directive_target_buf, slot, target);
        if let Some(p) = pos {
            webband_bridge::write_vec3(s, &s.agent_directive_pos_buf, slot, p);
        }
    }

    fn order_guard(&mut self) {
        let Some((id, _)) = self.selected() else { return };
        // The ward is the NEXT cycled ally — press G again to walk the ward
        // along the roster.
        let n = self.b.slot_map.len();
        if n < 2 {
            self.note = "there is no one else to guard".into();
            return;
        }
        self.ward_cur = (self.ward_cur + 1) % n;
        if self.ward_cur == self.sel {
            self.ward_cur = (self.ward_cur + 1) % n;
        }
        let (ward_id, ward_slot) = self.b.slot_map[self.ward_cur].clone();
        let target = self.stagger[ward_slot];
        self.set_directive(1, target, None);
        let (a, b) = (self.name_of(&id), self.name_of(&ward_id));
        self.note = format!("{a} guards {b}");
        println!("[orders] {a} guards {b} (ward stagger {target})");
        self.refresh_hud();
    }

    fn order_hold(&mut self) {
        let Some((id, slot)) = self.selected() else { return };
        // Hold anchors where they stand NOW — the Webband grammar exactly
        // (move first, then stamp it).
        let p = self.b.pos_at(slot);
        self.set_directive(2, 0, Some(p));
        let a = self.name_of(&id);
        self.note = format!("{a} holds ({:.0}, {:.0})", p[0], p[1]);
        println!("[orders] {a} holds at ({:.1}, {:.1})", p[0], p[1]);
        self.refresh_hud();
    }

    fn order_focus(&mut self) {
        let Some((id, _)) = self.selected() else { return };
        // The target pool is the STAGED cohort — there is nothing to focus on
        // when no one is at the fences, and saying so is better than writing a
        // directive that resolves to nothing.
        let cohort: Vec<usize> = match self.b.staged.as_ref() {
            Some(s) => s.cohort.clone(),
            None => Vec::new(),
        };
        let alive: Vec<usize> = {
            let n = self.b.w.n;
            let buf = self.b.state.agent_alive_buf.clone();
            let a = webband_bridge::read_u32(&mut self.b.state, &buf, n);
            cohort.into_iter().filter(|&r| a[r] == 1).collect()
        };
        if alive.is_empty() {
            self.note = "no raider is standing to focus on".into();
            println!("[orders] focus refused: no raider standing");
            self.refresh_hud();
            return;
        }
        self.foe_cur = (self.foe_cur + 1) % alive.len();
        let foe = alive[self.foe_cur];
        let target = self.stagger[foe];
        self.set_directive(3, target, None);
        let a = self.name_of(&id);
        self.note = format!("{a} focuses raider #{}", self.foe_cur + 1);
        println!("[orders] {a} focuses the raider at slot {foe} (stagger {target})");
        self.refresh_hud();
    }

    fn order_harry(&mut self) {
        let Some((id, _)) = self.selected() else { return };
        self.set_directive(4, 0, None);
        let a = self.name_of(&id);
        self.note = format!("{a} harries the wounded");
        println!("[orders] {a} harries");
        self.refresh_hud();
    }

    fn order_clear(&mut self) {
        let Some((id, _)) = self.selected() else { return };
        self.set_directive(0, 0, None);
        let a = self.name_of(&id);
        self.note = format!("{a} takes no orders");
        println!("[orders] {a} cleared");
        self.refresh_hud();
    }

    /// The COLONY-side standing order: which work this colonist reaches for
    /// first. Cycles chop -> forage -> hunt -> build -> cook -> craft -> haul
    /// -> grow -> no preference.
    fn order_trade(&mut self) {
        let Some((id, slot)) = self.selected() else { return };
        self.trade_cur = (self.trade_cur + 1) % (JOBS.len() + 1);
        let s = &self.b.state;
        let bufs = [
            s.agent_pri_chop_buf.clone(),
            s.agent_pri_forage_buf.clone(),
            s.agent_pri_hunt_buf.clone(),
            s.agent_pri_build_buf.clone(),
            s.agent_pri_cook_buf.clone(),
            s.agent_pri_craft_buf.clone(),
            s.agent_pri_haul_buf.clone(),
            s.agent_pri_grow_buf.clone(),
        ];
        for (i, buf) in bufs.iter().enumerate() {
            let v = if i == self.trade_cur { PRI_POINTS } else { 0.0 };
            webband_bridge::write_f32(&self.b.state, buf, slot, v);
        }
        let a = self.name_of(&id);
        let what = JOBS.get(self.trade_cur).copied().unwrap_or("whatever needs doing");
        self.note = format!("{a}: {what} first");
        println!("[orders] {a} works {what} first (pri +{PRI_POINTS})");
        self.refresh_hud();
    }

    // -- S13: the answer verb ----------------------------------------------

    fn answer(&mut self, choice: PetitionChoiceKind) {
        match self.b.answer(choice) {
            Ok(line) => {
                self.note = line.clone();
                println!("[politics] {line}");
            }
            Err(why) => {
                self.note = format!("cannot: {why}");
                println!("[politics] refused: {why}");
            }
        }
        self.refresh_hud();
        self.drain_log();
    }

    /// THE GUILD REPORT (`0`): everything the politics layer knows, as prose.
    fn politics_report(&mut self) {
        let c = &self.b.campaign;
        if !c.politics_enabled {
            println!("[guild] this campaign was founded WITHOUT the guild layer (--no-politics).");
            return;
        }
        println!("[guild] --- day {} ---", c.day);
        match &c.petition {
            Some(p) => {
                println!("[guild] THE ASK: {}", describe_petition(c, p));
                println!(
                    "[guild]   {} wants {} hands, {} rations, or {} gold — expires day {} \
                     (chosen: {:?})",
                    petitioner_name(c, p),
                    p.need_hands,
                    p.need_provisions,
                    p.need_gold,
                    p.expires_day,
                    p.chosen
                );
                for o in self.b.choices() {
                    println!(
                        "[guild]   [{}] {} — {}{}",
                        match o.choice {
                            PetitionChoiceKind::Send => "7",
                            PetitionChoiceKind::Pay => "8",
                            PetitionChoiceKind::Refuse => "9",
                        },
                        o.label,
                        o.detail,
                        o.blocked.map_or(String::new(), |w| format!("  BLOCKED: {w}"))
                    );
                }
            }
            None => println!("[guild] no one is asking anything of the guild today."),
        }
        let c = &self.b.campaign;
        // THE ASYMMETRIC DRIFT, made checkable: print the STAMP the ledger
        // actually holds (value, day) beside the value read TODAY. Nothing
        // decays on a pass — `standing_ledger.get` derives the drift from the
        // stamp at 0.5/day up from negative and 0.25/day down from positive,
        // and printing both halves lets a reader do that arithmetic.
        for (id, rec) in &c.standing_ledger.entries {
            let name = c
                .factions
                .iter()
                .find(|f| &f.id == id)
                .map_or(id.clone(), |f| f.name.clone());
            println!(
                "[guild] LEDGER {name}: stamped {:+.2} on day {} -> reads {:+.2} on day {}                  ({} days of drift)",
                rec.value,
                rec.day,
                c.standing_ledger.get(c.day, id),
                c.day,
                c.day - rec.day
            );
        }
        for f in &c.factions {
            let st = standing_with(c, &f.id);
            println!(
                "[guild] {:<24} {:>7.1} ({}){}{} — holds {} places",
                f.name,
                st,
                standing_word(st),
                if c.faction_ledger.is_hostile(&f.id) { "  HOSTILE" } else { "" },
                if petitioners(&c.factions).iter().any(|p| p.id == f.id) {
                    ""
                } else {
                    "  [asks nothing]"
                },
                f.hold_ids.len()
            );
        }
        match &c.ambition {
            Some(a) => {
                let (done, total) = ambition_progress(a);
                println!("[guild] THE ARC \"{}\": {done}/{total} stages closed", a.title);
                if let Some(s) = current_stage(a) {
                    println!("[guild]   now: {}", stage_line(c, s));
                }
            }
            None => println!("[guild] the founders carry no ambition this founding."),
        }
        for p in &c.afield {
            println!(
                "[guild] ON THE ROAD: {} — {} days out, {:.0} rations left",
                self.b.campaign.member_names(&p.member_ids),
                p.travel_days,
                p.provisions
            );
        }
        println!(
            "[guild] petitions opened {} / answered {} / lapsed {}",
            self.b.petitions_opened, self.b.petitions_answered, self.b.petitions_lapsed
        );
    }

    fn refresh_hud(&mut self) {
        let tick = self.b.tick;
        let day_min = tick % DAY_TICKS;
        self.hud.day = self.b.campaign.day as f32;
        self.hud.day_min = day_min as f32;
        // A Webband working day is 600 minutes of light — print it as a
        // 6:00-to-18:00 clock hour (the TS `timeOfDay` mapping, numerically).
        self.hud.hour = 6.0 + (day_min as f32) * 12.0 / DAY_TICKS as f32;
        self.hud.hands = self.b.slot_map.len() as f32;
        self.hud.gold = self.b.campaign.gold as f32;
        self.hud.renown = self.b.campaign.renown as f32;
        let inv = self
            .b
            .last_snap
            .as_ref()
            .map(|s| s.inventory.clone())
            .unwrap_or_default();
        let units = food_days(&inv.stacks);
        self.hud.food_units = units as f32;
        self.hud.food_days = (units / (self.b.slot_map.len().max(1) as f64)) as f32;
        self.hud.wealth =
            colony_wealth(self.b.campaign.gold, self.b.campaign.roster.len(), &inv) as f32;
        self.hud.raids = self.b.raids_staged as f32;
        self.hud.won = self.b.raids_won as f32;
        self.hud.lost = self.b.raids_lost as f32;
        self.hud.staged = u8::from(self.b.staged.is_some()) as f32;
        self.hud.speed = SPEEDS[self.speed_idx] as f32;
        self.hud.paused = u8::from(self.paused) as f32;
        self.hud.away = self.b.away.len() as f32;
        // One readback for the KO ledger and the live directive of the
        // selected colonist (both are custom fields, not `AgentView` columns).
        // Raiders standing come free with the frame's own snapshot — see
        // `agent_snapshot`.
        let n = self.b.w.n;
        let downed = {
            let buf = self.b.state.agent_downed_buf.clone();
            let d = webband_bridge::read_u32(&mut self.b.state, &buf, n);
            self.b.slot_map.iter().filter(|(_, slot)| d[*slot] == 1).count()
        };
        self.hud.downed = downed as f32;
        // The order WORD is read back from the fixture, not from a host-side
        // mirror: what the HUD claims and what the sim will act on are the
        // same value by construction.
        let (sel_name, order) = match self.selected() {
            Some((id, slot)) => {
                let buf = self.b.state.agent_directive_kind_buf.clone();
                let k = webband_bridge::read_u32(&mut self.b.state, &buf, n)[slot];
                let word = match k {
                    1 => "GUARD",
                    2 => "HOLD",
                    3 => "FOCUS",
                    4 => "HARRY",
                    _ => "no order",
                };
                let trade = JOBS.get(self.trade_cur).copied().unwrap_or("no preference");
                (
                    format!("{} (#{}/{})", self.name_of(&id), self.sel + 1, self.b.slot_map.len()),
                    format!("{word} · works {trade} first"),
                )
            }
            None => ("nobody".to_string(), "—".to_string()),
        };
        self.hud.texts.insert("wb_sel".into(), sel_name);
        self.hud.texts.insert("wb_order".into(), order);
        self.hud.texts.insert("wb_note".into(), self.note.clone());
        self.refresh_politics_texts();
        self.hud_at = tick;
    }

    fn refresh_politics_texts(&mut self) {
        let c = &self.b.campaign;
        if !c.politics_enabled {
            self.hud.texts.insert("wb_ask".into(), "(no guild layer — --no-politics)".into());
            self.hud.texts.insert("wb_powers".into(), "—".into());
            self.hud.texts.insert("wb_arc".into(), "—".into());
            return;
        }
        let ask = match &c.petition {
            Some(p) => {
                let left = p.expires_day - c.day;
                let blocked: Vec<String> = self
                    .b
                    .choices()
                    .into_iter()
                    .filter_map(|o| o.blocked.map(|w| format!("{:?}: {w}", o.choice)))
                    .collect();
                format!(
                    "{} asks {} hands / {} gold — {} day(s) left{}{}",
                    petitioner_name(c, p),
                    p.need_hands,
                    p.need_gold,
                    left.max(0),
                    p.chosen.map_or(String::new(), |ch| format!(" [{ch:?} taken]")),
                    if blocked.is_empty() {
                        String::new()
                    } else {
                        format!("  ({})", blocked.join("; "))
                    }
                )
            }
            None => "nobody is asking anything of the guild".to_string(),
        };
        let c = &self.b.campaign;
        let powers = c
            .factions
            .iter()
            .map(|f| {
                let st = standing_with(c, &f.id);
                format!(
                    "{} {:+.0}{}",
                    f.name.split(' ').next_back().unwrap_or(&f.name),
                    st,
                    if c.faction_ledger.is_hostile(&f.id) { "!" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("  ");
        let arc = match &c.ambition {
            Some(a) => {
                let (done, total) = ambition_progress(a);
                match current_stage(a) {
                    Some(s) => format!("{} [{done}/{total}] {}", a.title, stage_line(c, s)),
                    None => format!("{} — ACHIEVED", a.title),
                }
            }
            None => "no ambition this founding".to_string(),
        };
        self.hud.texts.insert("wb_ask".into(), ask);
        self.hud.texts.insert("wb_powers".into(), powers);
        self.hud.texts.insert("wb_arc".into(), arc);
    }

    /// Write the whole game state — host AND fixture — to `--save-dir`.
    fn save_all(&mut self) {
        let dir = self.save_dir.clone();
        match self.b.save_all(&dir) {
            Ok(r) => {
                self.saves += 1;
                println!(
                    "[webband_play] SAVED day {} tick {} -> {} ({} GPU buffers, {} KiB{})",
                    self.b.campaign.day,
                    self.b.tick,
                    dir.display(),
                    r.buffers,
                    r.bytes / 1024,
                    if r.skipped.is_empty() {
                        String::new()
                    } else {
                        format!("; {} buffers not copyable: {:?}", r.skipped.len(), r.skipped)
                    }
                );
            }
            Err(e) => println!("[webband_play] save failed: {e}"),
        }
    }

    /// The determinism digest, in the spine's own discipline: SIM-STATE
    /// buffers only (S5 finding 2 — `tally_*`/`count_*` views ride the
    /// engine's lossy fold window and are not run-to-run stable), plus the
    /// serialized campaign. This is what the save/resume proof compares.
    fn digest(&mut self) -> String {
        let fixture = sig_hash(&self.b.fixture_signature());
        let campaign = fnv1a(
            serde_json::to_string(&self.b.campaign)
                .expect("campaign serializes")
                .as_bytes(),
        );
        let log = fnv1a(self.b.log.join("\n").as_bytes());
        format!("fixture={fixture:#018x} campaign={campaign:#018x} log={log:#018x}")
    }

    fn report(&mut self) {
        self.drain_log();
        let c = &self.b.campaign;
        println!(
            "[webband_play] SUMMARY  day {}  gold {}  renown {}  roster {}  raids {} \
             (won {} / lost {})  chronicle {} entries  saves {}",
            c.day,
            c.gold,
            c.renown,
            c.roster.len(),
            self.b.raids_staged,
            self.b.raids_won,
            self.b.raids_lost,
            c.chronicle.len(),
            self.saves
        );
        for r in &self.b.raid_log {
            println!(
                "[webband_play] RAID day {} victory={} tier={} loot={} downed={} plunder={:?}",
                r.day, r.victory, r.tier, r.gold_looted, r.downed, r.plunder_taken
            );
        }
        if self.b.campaign.politics_enabled {
            println!(
                "[webband_play] POLITICS  petitions opened {} / answered {} / lapsed {}  \
                 stages closed {}  band notices {}  dispatch refusals {}",
                self.b.petitions_opened,
                self.b.petitions_answered,
                self.b.petitions_lapsed,
                self.b.stages.len(),
                self.b.band_notices.len(),
                self.b.refusals.len()
            );
            for (day, fid, kind, answer) in &self.b.petition_log {
                println!("[webband_play] ASK day {day} {fid} {kind} -> {answer}");
            }
            for s in &self.b.stages {
                println!("[webband_play] STAGE {s}");
            }
            for w in &self.b.refusals {
                println!("[webband_play] REFUSED A SEND: {w}");
            }
            self.politics_report();
        }
        for e in self.b.campaign.chronicle.iter() {
            println!(
                "[chronicle] day {:>3}  {}",
                e.day,
                e.headline.clone().unwrap_or_else(|| e.text.clone())
            );
        }
    }
}

impl PlayableRuntime for CampaignRuntime {
    fn tick(&self) -> u64 {
        u64::from(self.b.tick)
    }

    fn step(&mut self) {
        if self.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            println!(
                "[webband_play] --exit-after-secs reached at tick {} (day {}). Closing.",
                self.b.tick, self.b.campaign.day
            );
            self.report();
            // A bounded harness must never leave a window running; this is
            // the last thing the process does.
            std::process::exit(0);
        }
        if self.paused {
            return;
        }
        for _ in 0..SPEEDS[self.speed_idx] {
            self.b.step_one();
            if self.b.tick % DAY_TICKS == 0 {
                self.dawn();
            }
        }
        if self.b.tick.wrapping_sub(self.hud_at) >= HUD_PERIOD {
            self.refresh_hud();
            self.drain_log();
        }
    }

    fn set_input(&mut self, field: &str, value: f32) {
        match field {
            "host.pause" => {
                self.paused = !self.paused;
                // Refresh on the edge: a paused HUD must show the minute the
                // world actually stopped on, not the last cadence sample.
                self.refresh_hud();
                println!(
                    "[webband_play] {} at day {} minute {}",
                    if self.paused { "PAUSED" } else { "RESUMED" },
                    self.b.campaign.day,
                    self.b.tick % DAY_TICKS
                );
            }
            "host.speed" => {
                self.speed_idx = (value.round().max(0.0) as usize).min(SPEEDS.len() - 1);
                self.hud.speed = SPEEDS[self.speed_idx] as f32;
                println!("[webband_play] speed x{}", SPEEDS[self.speed_idx]);
            }
            "host.force_raid" => self.force_raid(),
            "host.save" => self.save_all(),
            "host.chronicle" => {
                println!(
                    "[webband_play] --- the chronicle, {} entries ---",
                    self.b.campaign.chronicle.len()
                );
                for e in self.b.campaign.chronicle.iter().rev().take(12) {
                    println!(
                        "  day {:>3}  {}",
                        e.day,
                        e.headline.clone().unwrap_or_else(|| e.text.clone())
                    );
                }
            }
            // S13 — the player's hands.
            "host.select_next" => self.select_next(),
            "host.order_guard" => self.order_guard(),
            "host.order_hold" => self.order_hold(),
            "host.order_focus" => self.order_focus(),
            "host.order_harry" => self.order_harry(),
            "host.order_clear" => self.order_clear(),
            "host.order_trade" => self.order_trade(),
            "host.answer_send" => self.answer(PetitionChoiceKind::Send),
            "host.answer_pay" => self.answer(PetitionChoiceKind::Pay),
            "host.answer_refuse" => self.answer(PetitionChoiceKind::Refuse),
            "host.report" => self.politics_report(),
            _ => {}
        }
    }

    fn agent_snapshot(&mut self) -> Vec<engine_play_api::AgentView> {
        let mut views = self.b.state.agent_snapshot();
        // Live raid state, free: the frame already paid for this readback.
        // The raid pool is pre-spawned DORMANT at the rim (S5's design), so
        // "raiders standing" must mean "bodies the staging woke that are
        // still alive", not "alive bodies of raider type".
        self.hud.raiders = match self.b.staged.as_ref() {
            Some(s) => s
                .cohort
                .iter()
                .filter(|&&r| views.get(r).is_some_and(|v| v.alive))
                .count() as f32,
            None => 0.0,
        };
        // S13 — HIGHLIGHT THE SELECTION. Presentation only: the mana sentinel
        // goes into this COPY of the views (the renderer's first matching
        // `AgentVisual` is the selection band prepended in
        // `selection_render_descriptor`), never into the sim.
        if let Some((_, slot)) = self.selected() {
            if let Some(v) = views.get_mut(slot) {
                v.mana = SEL_MANA;
            }
        }
        views
    }

    fn view_value(&mut self, view: &str, slot: u32) -> f32 {
        match self.hud.get(view) {
            Some(v) => v,
            // Anything else is the fixture's own materialized view.
            None => self.b.state.view_value(view, slot),
        }
    }

    fn view_text(&mut self, view: &str) -> Option<String> {
        self.hud.texts.get(view).cloned()
    }

    fn render_descriptor(&self) -> &'static str {
        self.render_desc
    }

    fn controls_descriptor(&self) -> &'static str {
        HOST_CONTROLS
    }

    fn ui_descriptor(&self) -> &'static str {
        // The HUD is built host-side (`campaign_ui_model`) because it prints
        // CAMPAIGN numbers, which no fixture `ui {}` block can name.
        r#"{"hud":[],"screens":[]}"#
    }
}

/// `--order-demo` — MEASURE THE PLAYER'S HANDS, headlessly, through the exact
/// path a keypress takes.
///
/// The claim under test is the one that justifies removing turns: an ORDER
/// changes what a colonist does, in real time, and the player can give it. So
/// this drives the campaign to a staged raid, then calls
/// `set_input("host.select_next" / "host.order_hold")` — the same two strings
/// the `Tab` and `H` key bindings resolve to, no private back door — and
/// samples positions for the rest of the fight:
///
///   * the HELD colonist's greatest distance from the anchor they were
///     standing on when the order landed, and
///   * an UNORDERED control colonist's greatest distance from where THEY were
///     standing at that same instant.
///
/// `raid_directives` (S5, re-measured at S5c) pins the same contrast inside
/// the test suite at 3.95 u held vs 9.54 u chasing. This is that measurement
/// taken by the shipped binary.
fn run_order_demo(bridge: Bridge, ticks: u32) {
    let mut rt = CampaignRuntime::new(bridge, 0, 0.0, Some(2), None);
    // Step until the raiders are AT THE FENCES AND MOVING — tick by tick, not
    // day by day: a raid musters at dawn+12 and can be over inside the same
    // day, so a day-granular wait watches the fight from the far side of it.
    let mut waited = 0u32;
    while waited < DAY_TICKS * 6 {
        rt.b.step_one();
        waited += 1;
        if rt.b.tick % DAY_TICKS == 0 {
            rt.dawn();
        }
        if rt
            .b
            .staged
            .as_ref()
            .is_some_and(|s| rt.b.tick > s.muster_tick + 40)
        {
            break;
        }
    }
    let staged = rt.b.staged.is_some();
    println!("[order-demo] raid staged: {staged} at tick {} (day {})", rt.b.tick, rt.b.campaign.day);
    if rt.b.slot_map.len() < 2 {
        println!("[order-demo] fewer than two colonists — nothing to compare.");
        return;
    }
    // THE ORDER, through the player's own input path.
    rt.sel = 0;
    let (held_id, held_slot) = rt.b.slot_map[0].clone();
    rt.set_input("host.order_hold", 1.0);
    let (ctrl_id, ctrl_slot) = rt.b.slot_map[1].clone();
    let anchor = rt.b.pos_at(held_slot);
    let ctrl_start = rt.b.pos_at(ctrl_slot);
    println!(
        "[order-demo] HELD {} at ({:.2}, {:.2}) | CONTROL {} at ({:.2}, {:.2}) — no order",
        rt.name_of(&held_id),
        anchor[0],
        anchor[1],
        rt.name_of(&ctrl_id),
        ctrl_start[0],
        ctrl_start[1]
    );
    let cohort: Vec<usize> = rt.b.staged.as_ref().map(|s| s.cohort.clone()).unwrap_or_default();
    let (mut held_max, mut ctrl_max, mut samples) = (0.0f32, 0.0f32, 0u32);
    let mut over_at = 0u32;
    for _ in 0..(ticks / 5) {
        for _ in 0..5 {
            rt.b.step_one();
        }
        // ONLY while the storm is running: a directive steers the defense
        // verbs, and once the raid settles everyone goes back to work and
        // walks wherever the work is — measuring past that measures nothing.
        let n = rt.b.w.n;
        let alive = {
            let b = rt.b.state.agent_alive_buf.clone();
            webband_bridge::read_u32(&mut rt.b.state, &b, n)
        };
        let ra = {
            let b = rt.b.state.agent_raid_active_buf.clone();
            webband_bridge::read_u32(&mut rt.b.state, &b, n)
        };
        let active = cohort.iter().any(|&r| alive[r] == 1 && ra[r] == 1);
        if !active {
            over_at = rt.b.tick;
            break;
        }
        let h = rt.b.pos_at(held_slot);
        let c = rt.b.pos_at(ctrl_slot);
        let dh = ((h[0] - anchor[0]).powi(2) + (h[1] - anchor[1]).powi(2)).sqrt();
        let dc = ((c[0] - ctrl_start[0]).powi(2) + (c[1] - ctrl_start[1]).powi(2)).sqrt();
        held_max = held_max.max(dh);
        ctrl_max = ctrl_max.max(dc);
        samples += 1;
    }
    println!(
        "[order-demo] over {samples} samples while the raid ran (it settled at tick {over_at}): \
         HELD ranged {held_max:.2} u from its anchor; the UNORDERED control ranged {ctrl_max:.2} u \
         from where it stood."
    );
    // A colonist that is DOWNED does not move either, so the "held" number is
    // only evidence if they were on their feet the whole time. Print it.
    {
        let n = rt.b.w.n;
        let d = { let b = rt.b.state.agent_downed_buf.clone(); webband_bridge::read_u32(&mut rt.b.state, &b, n) };
        let hp = { let b = rt.b.state.agent_hp_buf.clone(); webband_bridge::read_f32(&mut rt.b.state, &b, n) };
        println!(
            "[order-demo] at the end: HELD downed={} hp={:.0} | CONTROL downed={} hp={:.0}",
            d[held_slot], hp[held_slot], d[ctrl_slot], hp[ctrl_slot]
        );
    }
    println!(
        "[order-demo] VERDICT: {}",
        if held_max + 1.0 < ctrl_max {
            "the order changed what the colonist did."
        } else {
            "NO measurable difference — read the numbers, not this sentence."
        }
    );
}

/// A stage in the arc's own generated words. `describe_stage` wants the
/// pieces, not the campaign — this is the one place that gathers them.
fn stage_line(c: &webband_app::campaign::Campaign, s: &webband_app::ambition::AmbitionStage) -> String {
    describe_stage(&c.factions, &c.founding.world, &|id: &str| c.companion_display_name(id), s)
}

/// The fixture's own render descriptor with ONE band prepended: any agent
/// whose `mana` sits at [`SEL_MANA`] paints bright cyan. `agent_material`
/// takes the FIRST matching visual, so the selected colonist wins over the
/// `creature_type is Colonist` band beneath it and every other agent is
/// untouched. Leaked once (the trait wants `&'static str`); the fixture never
/// writes mana, so nothing else can land in the band by accident.
fn selection_render_descriptor(base: &'static str) -> &'static str {
    let sel = format!(
        r#"{{"when":{{"field":"mana","lo":{},"hi":{}}},"color":[60,255,255]}}"#,
        SEL_MANA - 0.5,
        SEL_MANA + 0.5
    );
    match base.find("\"agents\":[") {
        Some(i) => {
            let cut = i + "\"agents\":[".len();
            let mut out = String::with_capacity(base.len() + sel.len() + 1);
            out.push_str(&base[..cut]);
            out.push_str(&sel);
            if !base[cut..].starts_with(']') {
                out.push(',');
            }
            out.push_str(&base[cut..]);
            Box::leak(out.into_boxed_str())
        }
        None => base,
    }
}

/// The campaign HUD. Every `{key}` here is answered by
/// [`CampaignRuntime::view_value`] (numbers) or [`CampaignRuntime::view_text`]
/// (prose) through `PlayerConfig::hud_views` / `hud_texts` — the generic seams
/// S12 and S13 added to `engine_play`, so this file needs no renderer changes.
fn campaign_ui_model() -> UiModel {
    UiModel {
        hud: vec![
            Widget::Text {
                template: "WEBBAND   Day {wb_day}    hour {wb_hour}   (minute {wb_daymin} of 600)"
                    .into(),
            },
            Widget::Text {
                template: "colonists {wb_hands} (afield {wb_away})   gold {wb_gold}   \
                           renown {wb_renown}   wealth {wb_wealth}   \
                           larder {wb_food} units ({wb_fooddays}d/mouth)"
                    .into(),
            },
            Widget::Text {
                template: "RAID  raiders standing {wb_raiders}   colonists downed {wb_downed}   \
                           staged {wb_staged}   fought {wb_raids}  won {wb_won}  lost {wb_lost}"
                    .into(),
            },
            Widget::Text { template: "SELECTED  {wb_sel}  —  {wb_order}".into() },
            Widget::Text { template: "ASK  {wb_ask}".into() },
            Widget::Text { template: "POWERS  {wb_powers}".into() },
            Widget::Text { template: "ARC  {wb_arc}".into() },
            Widget::Text { template: "\u{203a} {wb_note}".into() },
            Widget::Text {
                template: "speed x{wb_speed}   paused {wb_paused}   \
                           [space] pause  [1-4] speed  [R] raid  [S] save  [C] chronicle"
                    .into(),
            },
            Widget::Text {
                template: "[Tab] select  [G] guard  [H] hold  [F] focus  [Y] harry  \
                           [X] clear  [V] trade   |   [7] send  [8] pay  [9] refuse  \
                           [0] guild report"
                    .into(),
            },
        ],
        screens: vec![],
    }
}

// ---------------------------------------------------------------------------

struct Args {
    campaign_seed: u32,
    scenario: ScenarioId,
    sign_bands: usize,
    speed_idx: usize,
    exit_after_secs: f32,
    headless_days: Option<usize>,
    force_raid_day: Option<i64>,
    shots_dir: Option<String>,
    /// Where `[S]` and a finished `--headless` run write the two-part save.
    save_dir: Option<String>,
    /// Resume from a save directory instead of founding a new campaign.
    resume: Option<String>,
    /// Write the run's end-state determinism digest here (and to stdout).
    digest: Option<String>,
    /// S13: the guild layer, live by default.
    politics: bool,
    /// S13: the headless answer POLICY (send / pay / refuse / hold).
    petition_answer: Option<PetitionChoiceKind>,
    /// S13: `--order-demo N` — headless proof that an ORDER moves a colonist.
    order_demo: Option<u32>,
}

/// Valueless switches — everything else is a `--key value` pair.
const FLAGS: [&str; 2] = ["politics", "no-politics"];

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut map: HashMap<String, String> = HashMap::new();
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].trim_start_matches("--").to_string();
        if FLAGS.contains(&k.as_str()) {
            map.insert(k, "1".to_string());
            i += 1;
            continue;
        }
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        map.insert(k, v);
        i += 2;
    }
    let num = |k: &str| map.get(k).and_then(|v| v.parse::<f64>().ok());
    Args {
        campaign_seed: num("campaign-seed").map_or(CAMPAIGN_SEED, |v| v as u32),
        scenario: match map.get("scenario").map(String::as_str) {
            Some("town") => ScenarioId::Town,
            Some("wilderness") => ScenarioId::Wilderness,
            Some("city") => ScenarioId::City,
            _ => ScenarioId::Village,
        },
        // The bridge's own staging default (sign every band that fits the 20
        // colonist bodies) — the soak and the spine both pass 99.
        sign_bands: num("sign-bands").map_or(99, |v| v as usize),
        speed_idx: num("speed").map_or(1, |v| (v as usize).min(SPEEDS.len() - 1)),
        exit_after_secs: num("exit-after-secs").map_or(0.0, |v| v as f32),
        headless_days: num("headless").map(|v| v as usize),
        force_raid_day: num("force-raid-day").map(|v| v as i64),
        shots_dir: map.get("shots-dir").cloned(),
        save_dir: map.get("save-dir").cloned(),
        resume: map.get("resume").cloned(),
        digest: map.get("digest").cloned(),
        // DEFAULT-ON, opt-out. See the module header for why.
        politics: !map.contains_key("no-politics"),
        order_demo: num("order-demo").map(|v| v as u32),
        petition_answer: match map.get("petition-answer").map(String::as_str) {
            Some("send") => Some(PetitionChoiceKind::Send),
            Some("pay") => Some(PetitionChoiceKind::Pay),
            Some("refuse") => Some(PetitionChoiceKind::Refuse),
            _ => None, // "hold" / absent: nobody answers, and the deadline bites
        },
    }
}

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    eprintln!(
        "[webband_play] founding: campaign seed {} scenario {:?} politics {} | fixture seed \
         {SEED:#x} agents {AGENTS}",
        args.campaign_seed, args.scenario, args.politics
    );
    let bridge = match &args.resume {
        // D2: pick a campaign back up where it was left — host state AND the
        // colony's live GPU state (positions, larder, beliefs, staged raid).
        Some(dir) => {
            let (b, r) = Bridge::load_all(std::path::Path::new(dir))
                .unwrap_or_else(|e| panic!("resume from {dir}: {e}"));
            eprintln!(
                "[webband_play] RESUMED {dir}: day {} tick {} ({} GPU buffers restored{}{})",
                b.campaign.day,
                b.tick,
                r.buffers,
                if r.unplaced.is_empty() {
                    String::new()
                } else {
                    format!("; {} saved names with no live buffer", r.unplaced.len())
                },
                if r.missing.is_empty() {
                    String::new()
                } else {
                    format!("; {} live buffers not in the save", r.missing.len())
                }
            );
            b
        }
        None => match Bridge::new_with(
            args.scenario,
            args.campaign_seed,
            args.sign_bands,
            true,
            args.politics,
        ) {
            Some(b) => b,
            None => {
                eprintln!("[webband_play] no wgpu adapter — cannot construct the colony runtime.");
                std::process::exit(2);
            }
        },
    };
    eprintln!(
        "[webband_play] \"{}\": roster {} seated on colonist bodies ({} pooled), gold {}, \
         powers {}, arc {:?}",
        bridge.campaign.founding.name,
        bridge.campaign.roster.len(),
        bridge.free_slots.len(),
        bridge.campaign.gold,
        bridge.campaign.factions.len(),
        bridge.campaign.ambition.as_ref().map(|a| a.title.clone())
    );

    if let Some(ticks) = args.order_demo {
        run_order_demo(bridge, ticks);
        return Ok(());
    }

    // Headless: no window, no renderer — the campaign at full speed. Exists
    // to demonstrate that the real-time split (`step_one` + `dawn`) is the
    // batch loop, and to produce a log without a display.
    if let Some(days) = args.headless_days {
        let mut rt =
            CampaignRuntime::new(bridge, 0, 0.0, args.force_raid_day, args.petition_answer);
        if let Some(dir) = &args.save_dir {
            rt.save_dir = PathBuf::from(dir);
        }
        for _ in 0..days {
            for _ in 0..DAY_TICKS {
                rt.b.step_one();
            }
            rt.dawn();
        }
        rt.report();
        if args.save_dir.is_some() {
            rt.save_all();
        }
        if let Some(path) = &args.digest {
            let d = rt.digest();
            println!("[webband_play] DIGEST day {} tick {} {d}", rt.b.campaign.day, rt.b.tick);
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, format!("day {} tick {} {d}\n", rt.b.campaign.day, rt.b.tick))?;
        }
        return Ok(());
    }

    eprintln!(
        "[webband_play] render descriptor (selection band prepended): {}",
        selection_render_descriptor(bridge.state.render_descriptor())
    );

    if let Some(dir) = &args.shots_dir {
        let _ = std::fs::create_dir_all(dir);
        eprintln!("[webband_play] screenshots expected in {dir} (captured OS-side).");
    }

    let mut rt = CampaignRuntime::new(
        bridge,
        args.speed_idx,
        args.exit_after_secs,
        args.force_raid_day,
        args.petition_answer,
    );
    if let Some(dir) = &args.save_dir {
        rt.save_dir = PathBuf::from(dir);
    }

    let cfg = PlayerConfig {
        hud_views: Hud::keys(),
        hud_texts: Hud::text_keys(),
        ..PlayerConfig::default()
    };
    let player = Player::new(Box::new(rt), cfg, campaign_ui_model())?;
    player.run()
}
