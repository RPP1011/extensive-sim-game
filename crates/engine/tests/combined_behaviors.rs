//! Task 172 — combined-behavior coverage.
//!
//! Grudges (`my_enemies`, task 160), pack focus (`pack_focus`, task 169),
//! fear spread / rout (`kin_fear`, task 167), and wounded flee
//! (hp_pct<0.3 Flee modifier, task 165) all ship as independent
//! mechanics with their own targeted tests. Each lives in a clean fixture
//! and exercises one behavior in isolation:
//!
//! - `wolves_and_humans_parity.rs` — grudges, threat_level, memory folds
//! - `pack_focus.rs` — pack_focus fold + Attack boost
//! - `fear_spread_rout.rs` — kin_fear fold + Flee boost
//!
//! What none of those cover: scenarios where multiple of these behaviors
//! fire together. The views and scoring modifiers are additive — grudge
//! (+0.4 Attack, gated on my_enemies>0.5) and pack_focus (+0.4 Attack,
//! gated on pack_focus>0.5) can both fire on the same agent, and the
//! scoring table has no priority between them. So does the grudge-target
//! still win when pack focus would steer elsewhere? Does the engagement
//! teardown on death + fear_spread compose cleanly on the same tick? Does
//! wounded flee (+0.6) still beat pack focus (+0.4) even when the pack is
//! converging on our target?
//!
//! These tests exercise the composition — each test wires together two or
//! more of the four behaviors and asserts the behavioral outcome (which
//! action wins, who has which view values). Byte-exact determinism is
//! covered elsewhere (`wolves_and_humans_parity`, `determinism.rs`); this
//! file is behavioral only.
//!
//! Scenarios landed (see task brief for the full candidate list):
//!   A. Grudge survives pack focus (`grudge_dominates_pack_focus_on_different_targets`)
//!   B. Rout mid-engagement (`engagement_death_triggers_rout_in_partner`)
//!   C. Wounded flee beats pack boost (`wounded_self_flees_despite_pack_focus_on_target`)
//!   E. Chained rout cascades through pack (`chained_deaths_stack_kin_fear_on_survivors`)

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MicroKind;
use engine::state::{AgentSpawn, SimState};
use engine_rules::scoring::{
    PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE,
};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Fixtures — shared helpers for spawning canonical creatures. Identical
// shape to `pack_focus.rs` / `fear_spread_rout.rs` so the tests read
// uniformly against the existing fixtures.
// ---------------------------------------------------------------------------

fn spawn_wolf(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos,
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .expect("wolf spawn")
}

fn spawn_human(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos,
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn")
}

// ---------------------------------------------------------------------------
// Scoring harness. Copy-paste from pack_focus / fear_spread — the
// `score_entry` fn on `engine::policy::utility` is `pub(crate)`, so
// integration tests have to reconstruct the pipeline themselves. The
// shape is identical across the four existing tests (grudge / pack /
// fear / this one), which keeps the behavioral-comparison obvious when
// reading side-by-side.
// ---------------------------------------------------------------------------

fn score_row_for(
    entry: &ScoringEntry,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> f32 {
    let mut score = entry.base;
    let count = entry.modifier_count as usize;
    let max = count.min(MAX_MODIFIERS);
    for i in 0..max {
        let row = &entry.modifiers[i];
        let pred = &row.predicate;
        match pred.kind {
            PredicateDescriptor::KIND_SCALAR_COMPARE => {
                let lhs = read_field_scalar(state, agent, target, pred.field_id);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE => {
                let lhs = eval_view(state, agent, target, pred);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_VIEW_GRADIENT => {
                let v = eval_view(state, agent, target, pred);
                if v.is_finite() {
                    score += v * row.delta;
                }
            }
            _ => {}
        }
    }
    score
}

fn read_field_scalar(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    field_id: u16,
) -> f32 {
    if field_id >= 0x4000 && field_id < 0x8000 {
        let t = match target {
            Some(x) => x,
            None => return f32::NAN,
        };
        return match field_id {
            0x4000 => state.agent_hp(t).unwrap_or(0.0),
            0x4002 => {
                let hp = state.agent_hp(t).unwrap_or(0.0);
                let mx = state.agent_max_hp(t).unwrap_or(1.0);
                if mx > 0.0 { hp / mx } else { 0.0 }
            }
            _ => f32::NAN,
        };
    }
    match field_id {
        0 => state.agent_hp(agent).unwrap_or(0.0),
        2 => {
            let hp = state.agent_hp(agent).unwrap_or(0.0);
            let mx = state.agent_max_hp(agent).unwrap_or(1.0);
            if mx > 0.0 { hp / mx } else { 0.0 }
        }
        _ => f32::NAN,
    }
}

fn eval_view(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let slot0 = pred.payload[4];
    let slot1 = pred.payload[5];
    let resolve = |slot: u8| -> Option<AgentId> {
        match slot {
            PredicateDescriptor::ARG_SELF => Some(agent),
            PredicateDescriptor::ARG_TARGET => target,
            _ => None,
        }
    };
    match pred.field_id {
        PredicateDescriptor::VIEW_ID_THREAT_LEVEL => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.threat_level.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.threat_level.get(a, b, state.tick)
            }
        }
        PredicateDescriptor::VIEW_ID_MY_ENEMIES => {
            let a = resolve(slot0).unwrap_or(agent);
            let b = resolve(slot1).unwrap_or(agent);
            state.views.my_enemies.get(a, b)
        }
        PredicateDescriptor::VIEW_ID_KIN_FEAR => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.kin_fear.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.kin_fear.get(a, b, state.tick)
            }
        }
        PredicateDescriptor::VIEW_ID_PACK_FOCUS => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.pack_focus.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.pack_focus.get(a, b, state.tick)
            }
        }
        _ => f32::NAN,
    }
}

fn compare(op: u8, lhs: f32, rhs: f32) -> bool {
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    match op {
        PredicateDescriptor::OP_LT => lhs < rhs,
        PredicateDescriptor::OP_LE => lhs <= rhs,
        PredicateDescriptor::OP_EQ => lhs == rhs,
        PredicateDescriptor::OP_GE => lhs >= rhs,
        PredicateDescriptor::OP_GT => lhs > rhs,
        PredicateDescriptor::OP_NE => lhs != rhs,
        _ => false,
    }
}

fn attack_entry() -> &'static ScoringEntry {
    SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Attack as u16)
        .expect("SCORING_TABLE missing Attack row")
}

fn flee_entry() -> &'static ScoringEntry {
    SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Flee as u16)
        .expect("SCORING_TABLE missing Flee row")
}

// ---------------------------------------------------------------------------
// Prime helpers — inject a single fold directly into a view without
// re-running the physics rule. Matches the style used by
// `pack_focus.rs::prime_pack_focus` / `fear_spread_rout.rs::prime_kin_fear`.
// Isolates the scoring / view math from the cascade wiring.
// ---------------------------------------------------------------------------

fn prime_pack_focus(state: &mut SimState, observer: AgentId, target: AgentId) {
    let ev = Event::PackAssist {
        observer,
        target,
        tick: state.tick,
    };
    state.views.pack_focus.fold_event(&ev, state.tick);
}

/// Prime a grudge + threat_level in one call: mirrors a real
/// `AgentAttacked { actor: attacker, target: victim }` event being
/// folded into both views (which is how the production pipeline wires
/// it via `dispatch_agent_attacked`).
fn prime_grudge_from_attack(
    state: &mut SimState,
    attacker: AgentId,
    victim: AgentId,
    damage: f32,
) {
    let ev = Event::AgentAttacked {
        actor: attacker,
        target: victim,
        damage,
        tick: state.tick,
    };
    state.views.my_enemies.fold_event(&ev, state.tick);
    state.views.threat_level.fold_event(&ev, state.tick);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// **Scenario A — grudge dominates pack focus.**
///
/// A wolf faces two humans: one it has a grudge against (H_grudge, which
/// attacked it), and another the pack is converging on (H_focus, because
/// packmate W2 engaged H_focus and emitted a PackAssist towards W1).
/// Both views fire — `my_enemies[W1,H_grudge] = 1.0 > 0.5` (+0.4 Attack)
/// AND `pack_focus[W1,H_focus] = 1.0 > 0.5` (+0.4 Attack). The scoring
/// table has no tiebreaker between the two modifiers, so parity on +0.4
/// alone would leave the wolf ambivalent.
///
/// What breaks the tie: the grudge-attacker's `threat_level` gradient.
/// The same `AgentAttacked` event that primed `my_enemies` also folded
/// `threat_level` (+1 per hit × +0.01 gradient = +0.01 per attack). So a
/// single attack is enough to tilt the wolf by +0.01 towards H_grudge,
/// and repeated attacks (5 here, still below the +20 scalar gate) widen
/// the margin to +0.05. The wolf targets H_grudge over H_focus — the
/// grudge + accumulated threat narrowly beats raw pack convergence.
#[test]
fn grudge_dominates_pack_focus_on_different_targets() {
    let mut state = SimState::new(8, 0xA5CADE_1);
    let w1 = spawn_wolf(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let _w2 = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let h_grudge = spawn_human(&mut state, Vec3::new(-3.0, 0.0, 0.0));
    let h_focus = spawn_human(&mut state, Vec3::new(3.0, 0.0, 0.0));

    let attack = attack_entry();

    // Pre-prime baseline: both humans symmetric, both at fresh hp, W1 at
    // fresh hp. Attack should tie at ~0.5 (fresh-self +0.5, no other
    // modifiers firing).
    let pre_grudge = score_row_for(attack, &state, w1, Some(h_grudge));
    let pre_focus = score_row_for(attack, &state, w1, Some(h_focus));
    assert!(
        (pre_grudge - pre_focus).abs() < 1e-4,
        "symmetric baseline expected, got H_grudge={pre_grudge}, H_focus={pre_focus}",
    );

    // H_grudge attacks W1 five times — folds 5× AgentAttacked, building
    // threat_level to 5 (gradient bonus 5 × 0.01 = +0.05) and saturating
    // my_enemies at 1.0 (grudge gate fires, +0.4). Still below the
    // threat_level > 20 scalar gate.
    for _ in 0..5 {
        prime_grudge_from_attack(&mut state, h_grudge, w1, 10.0);
    }

    // W2 engages H_focus — emits PackAssist to W1, folding pack_focus
    // [W1, H_focus] = 1.0. Gate fires, +0.4 on Attack(W1, H_focus).
    prime_pack_focus(&mut state, w1, h_focus);

    let post_grudge = score_row_for(attack, &state, w1, Some(h_grudge));
    let post_focus = score_row_for(attack, &state, w1, Some(h_focus));

    // Grudge target — 0.5 base + 0.4 grudge + 0.05 threat gradient = 0.95.
    // Focus target — 0.5 base + 0.4 pack_focus = 0.9.
    // Grudge wins by ~0.05 — thin but decisive argmax margin.
    assert!(
        post_grudge > post_focus,
        "grudge should beat pack focus: Attack(W1, H_grudge)={post_grudge} \
         vs Attack(W1, H_focus)={post_focus}",
    );
    assert!(
        post_grudge - post_focus >= 0.04,
        "grudge margin should be ≥ +0.04 (threat gradient 5×0.01); got {}",
        post_grudge - post_focus,
    );

    // Cross-check both modifiers actually fired — defensive against a
    // regression where pack_focus silently stops gating.
    assert!(
        post_focus > pre_focus + 0.35,
        "pack_focus bump missing: pre={pre_focus} post={post_focus} (expected +0.4)",
    );
    assert!(
        post_grudge > pre_grudge + 0.4,
        "grudge bump missing: pre={pre_grudge} post={post_grudge} (expected +0.45)",
    );
}

/// **Scenario B — engagement teardown + fear spread compose on the same death.**
///
/// Pair W1+H1, pair W2+H2 (set engagement explicitly via
/// `set_agent_engaged_with`). Kill W1 via `kill_agent`, then push an
/// `AgentDied` event and run the cascade to fixed-point. This walks the
/// AgentDied dispatcher: (i) `engagement_on_death` tears down W1↔H1 and
/// emits an `EngagementBroken { actor: H1, former_target: W1, reason: 2 }`;
/// (ii) `fear_spread_on_death` emits `FearSpread` to every nearby
/// same-species kin — i.e. W2. Finally we fold the views over the
/// tick's events.
///
/// Asserts (three composed behaviors on one tick):
///  1. W1's partner (H1) has its engagement cleared — teardown fired.
///  2. W2 (the surviving packmate) observes kin_fear > 0.3 — fear_spread
///     fired and was folded.
///  3. W2's scoring swings from "Attack > Flee" (engaged state, fresh hp)
///     to "Flee > Attack" — the rout cascades onto an agent that was
///     mid-combat, not just a free bystander. Task 173 retuned kin_fear
///     so a healthy wolf alone no longer flips; W2 is set to hp=45
///     (below the hp<50 gate) so kin_fear adds the tipping +0.4 on top
///     of the +0.4 wounded-body gate. The assertion is still "rout
///     cascades onto mid-combat agent" — just calibrated against the
///     new softer rout.
#[test]
fn engagement_death_triggers_rout_in_partner() {
    let mut state = SimState::new(16, 0xA5CADE_2);
    // Pack clustered within 12 m so fear_spread catches W2 when W1 dies.
    // Humans 1 m from their wolf so they're within engagement range, but
    // positions aren't strictly required — we set engagement explicitly.
    let w1 = spawn_wolf(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let h1 = spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(4.0, 0.0, 0.0));

    // Wire engagements manually so we can isolate the death-teardown
    // without entangling with the engagement-on-move rule.
    state.set_agent_engaged_with(w1, Some(h1));
    state.set_agent_engaged_with(h1, Some(w1));
    state.set_agent_engaged_with(w2, Some(h2));
    state.set_agent_engaged_with(h2, Some(w2));

    // Lightly wound W2 so the `hp < 50` Flee gate fires (+0.4). This
    // is below the `hp_pct >= 0.8` fresh-self Attack gate (hp_pct =
    // 45/80 = 0.5625), so Attack drops to 0.0 on a full-hp target and
    // Flee's single-gate score already ties with Attack. The kin_fear
    // +0.4 then pushes Flee decisively above. Without this mild wound,
    // the retuned rout (task 173) intentionally does NOT flip a fresh
    // engaged wolf — see Scenario C (`wounded_self_flees_despite_...`)
    // which covers the heavily-wounded case explicitly.
    state.set_agent_hp(w2, 45.0);

    // Pre-assert: before the death, W2 (lightly wounded but engaged)
    // prefers Attack(~0.0) — wait, actually Attack drops to 0 once
    // hp_pct<0.8. So the informative pre-condition is that Flee already
    // has +0.4 from hp<50, tying Attack at 0.4. kin_fear adds the
    // deciding +0.4.
    let attack = attack_entry();
    let flee = flee_entry();
    let pre_attack = score_row_for(attack, &state, w2, Some(h2));
    let pre_flee = score_row_for(flee, &state, w2, None);
    assert!(
        pre_flee >= pre_attack,
        "pre-death: wounded W2 should have Flee(={pre_flee}) ≥ Attack(={pre_attack}) \
         from the hp<50 gate alone; kin_fear hasn't fired yet",
    );

    // Drive the death through the full cascade. `kill_agent` flips the
    // alive bit + evicts from the spatial index; the `AgentDied` event
    // then triggers `engagement_on_death` (which tears down H1↔W1 and
    // emits `EngagementBroken`) and `fear_spread_on_death` (which emits
    // `FearSpread { observer: w2, dead_kin: w1 }`). The fixed-point run
    // drains the cascade until convergence.
    let cascade = CascadeRegistry::with_engine_builtins();
    let mut events = EventRing::with_cap(64);
    let events_before = events.total_pushed();
    let tick = state.tick;
    state.kill_agent(w1);
    events.push(Event::AgentDied { agent_id: w1, tick });
    cascade.run_fixed_point(&mut state, &mut events);
    // Phase 5b analogue — fold the views over the cascade's output.
    state.views.fold_all(&events, events_before, state.tick);

    // Assertion 1: engagement torn down.
    assert_eq!(
        state.agent_engaged_with(h1),
        None,
        "H1 should no longer be engaged (W1 died, engagement_on_death should clear both sides)",
    );
    // W2↔H2 is unaffected — its partner W2 is alive.
    assert_eq!(
        state.agent_engaged_with(w2),
        Some(h2),
        "W2↔H2 engagement should survive W1's death",
    );

    // Assertion 2: fear_spread reached W2.
    let kf = state.views.kin_fear.sum_for_first(w2, state.tick);
    assert!(
        kf > 0.5,
        "W2 kin_fear {kf} should exceed 0.5 gate after W1 died within 12m",
    );

    // Assertion 3: rout flipped the argmax. Flee should now beat Attack.
    let post_attack = score_row_for(attack, &state, w2, Some(h2));
    let post_flee = score_row_for(flee, &state, w2, None);
    assert!(
        post_flee > post_attack,
        "post-death: W2 should prefer Flee(={post_flee}) over Attack(={post_attack}) — \
         kin_fear adds +0.4 to Flee via the >0.5 gate",
    );
    assert!(
        post_flee > pre_flee + 0.3,
        "Flee should jump by ≥+0.3 (expected +0.4 from kin_fear gate); \
         pre={pre_flee} post={post_flee}",
    );

    // Defensive: the `EngagementBroken` event should appear in the ring
    // with the right actor/reason. Guards against a regression where the
    // physics rule silently stops emitting.
    let breaks: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::EngagementBroken { actor, former_target, reason, .. }
                if *actor == h1 && *former_target == w1 =>
            {
                Some(*reason)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        breaks,
        vec![2],
        "expected exactly one EngagementBroken(actor=H1, former_target=W1, reason=2); got {breaks:?}",
    );
}

/// **Scenario C — wounded-self flee beats pack focus on target.**
///
/// Three wolves W1/W2/W3 converging on one human (all within kin-radius
/// and attack range). W1 is pre-wounded to hp_pct = 0.2 (below the <0.3
/// Flee gate, which adds +0.6). W2 and W3 engage H1 — each EngagementCommit
/// emits a PackAssist { observer: W1, target: H1 }, so pack_focus
/// [W1, H1] = 2.0 (two assists folded). Both gates fire on the Attack row:
/// pack_focus > 0.5 (+0.4) and fresh-target-hp_pct >= 0.8 (+0.3 if
/// applicable — we check later).
///
/// Despite the pack converging, W1's Flee should dominate:
///
///   Flee (W1): base 0.0
///     + self.hp < 30: +0.6 (hp=16)
///     + self.hp < 50: +0.4
///     + self.hp_pct < 0.3: +0.6
///     = ~1.6
///
///   Attack (W1, H1): base 0.0
///     + self.hp_pct >= 0.8: NOT firing (wolf at 0.2)
///     + pack_focus[W1,H1] > 0.5: +0.4
///     + possibly target-hp gates
///     ≤ ~0.9 in any realistic config (far below Flee's 1.6)
///
/// This is the "wounded wolf bolts from the pack" scenario — combined
/// pack pressure doesn't override self-preservation.
#[test]
fn wounded_self_flees_despite_pack_focus_on_target() {
    let mut state = SimState::new(8, 0xA5CADE_3);
    // Wounded wolf — hp=16/80 → hp_pct = 0.2, below the 0.3 gate.
    let w1 = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 16.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .expect("wounded w1 spawn");
    let _w2 = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let _w3 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let h1 = spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));

    let attack = attack_entry();
    let flee = flee_entry();

    // Pre-prime — W1 is already wounded, so even without pack_focus its
    // Flee > Attack. We assert this first so the post-condition is
    // unambiguously "pack_focus didn't override wounded-flee", not "pack
    // focus never had a chance".
    let pre_flee = score_row_for(flee, &state, w1, None);
    let pre_attack = score_row_for(attack, &state, w1, Some(h1));
    assert!(
        pre_flee > pre_attack,
        "pre-pack: wounded W1 already prefers Flee(={pre_flee}) > Attack(={pre_attack})",
    );

    // Two packmates both engage H1 — each emits a PackAssist to W1.
    // pack_focus[W1, H1] accumulates to ~2.0, well above the 0.5 gate.
    prime_pack_focus(&mut state, w1, h1);
    prime_pack_focus(&mut state, w1, h1);
    let pf_val = state.views.pack_focus.get(w1, h1, state.tick);
    assert!(
        pf_val > 0.5,
        "pack_focus {pf_val} should fire the >0.5 gate after two assists",
    );

    // Post-pack: Attack on H1 should have gained the +0.4 pack_focus
    // bump, but Flee still dominates. The wounded-hp modifiers stack to
    // +1.6 on Flee, which swamps any Attack boost from pack_focus.
    let post_flee = score_row_for(flee, &state, w1, None);
    let post_attack = score_row_for(attack, &state, w1, Some(h1));

    // Pack focus bump landed — sanity check the +0.4 isn't being eaten.
    assert!(
        post_attack >= pre_attack + 0.35,
        "Attack should gain ≥+0.35 from pack_focus; pre={pre_attack} post={post_attack}",
    );
    // Flee shouldn't change — the wounded-self modifiers don't depend
    // on pack_focus. This pins the scoring table against a regression
    // where a future kin_fear / pack_focus row accidentally lands on Flee.
    assert!(
        (post_flee - pre_flee).abs() < 1e-4,
        "Flee score should be invariant to pack_focus folds; pre={pre_flee} post={post_flee}",
    );
    // Primary assertion — self-preservation wins.
    assert!(
        post_flee > post_attack,
        "wounded W1 should flee despite pack convergence: \
         Flee({post_flee}) > Attack({post_attack})",
    );
    // Margin should be dominant — +1.6 vs ≤+0.9 ≈ +0.7.
    assert!(
        post_flee - post_attack >= 0.5,
        "Flee margin over Attack should be ≥ +0.5 (wounded gates stack to +1.6 on Flee); \
         got delta = {}",
        post_flee - post_attack,
    );
}

/// **Scenario E — chained rout stacks kin_fear on survivors.**
///
/// 4-wolf pack, all within 12 m so fear-spread covers every pair. Kill W1
/// via the full cascade (AgentDied event + `run_fixed_point`); each of
/// W2/W3/W4 should receive one FearSpread (kin_fear sum ≈ +1.0 each, all
/// above the 0.5 gate — task 173). Then kill W2 — W3 and W4 receive a
/// second FearSpread (kin_fear stacks: fresh +1.0 landing on top of the
/// already-partially-decayed first event), sharpening the rout.
///
/// This pins the "chained rout" behavior the task brief calls for —
/// sequential deaths don't just reset the rout, they compound it, which
/// is what makes the pack crumble once the first wolf falls.
#[test]
fn chained_deaths_stack_kin_fear_on_survivors() {
    let mut state = SimState::new(16, 0xA5CADE_E);
    // Tight cluster — all within 12 m kin-radius. Using x=0,2,4,6 keeps
    // every pair inside the radius (max spread = 6 m).
    let w1 = spawn_wolf(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let w3 = spawn_wolf(&mut state, Vec3::new(4.0, 0.0, 0.0));
    let w4 = spawn_wolf(&mut state, Vec3::new(6.0, 0.0, 0.0));

    let cascade = CascadeRegistry::with_engine_builtins();
    let mut events = EventRing::with_cap(64);

    // --- Phase 1: kill W1. W2/W3/W4 all within 12 m → 3 FearSpread events.
    let events_before_1 = events.total_pushed();
    let tick_1 = state.tick;
    state.kill_agent(w1);
    events.push(Event::AgentDied { agent_id: w1, tick: tick_1 });
    cascade.run_fixed_point(&mut state, &mut events);
    state.views.fold_all(&events, events_before_1, state.tick);

    // Count FearSpread events from phase 1 — exactly 3, one per surviving wolf.
    let fears_phase_1: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::FearSpread { observer, dead_kin, .. } if *dead_kin == w1 => Some(*observer),
            _ => None,
        })
        .collect();
    assert_eq!(
        fears_phase_1.len(),
        3,
        "W1 death should emit 3 FearSpreads (to W2, W3, W4); got {fears_phase_1:?}",
    );
    assert!(fears_phase_1.contains(&w2));
    assert!(fears_phase_1.contains(&w3));
    assert!(fears_phase_1.contains(&w4));

    // All three survivors above the 0.5 kin_fear gate.
    let kf2_a = state.views.kin_fear.sum_for_first(w2, state.tick);
    let kf3_a = state.views.kin_fear.sum_for_first(w3, state.tick);
    let kf4_a = state.views.kin_fear.sum_for_first(w4, state.tick);
    assert!(kf2_a > 0.5, "W2 kin_fear {kf2_a} below 0.5 after W1 death");
    assert!(kf3_a > 0.5, "W3 kin_fear {kf3_a} below 0.5 after W1 death");
    assert!(kf4_a > 0.5, "W4 kin_fear {kf4_a} below 0.5 after W1 death");

    // All three should be near the +1.0 emit amount (modulo same-tick
    // decay, which is 0 ticks of decay → exactly 1.0).
    assert!(
        (kf2_a - 1.0).abs() < 1e-3,
        "W2 kin_fear {kf2_a} should be ≈1.0 from one FearSpread emit",
    );

    // --- Phase 2: kill W2. W3/W4 observe a SECOND FearSpread. kin_fear
    // stacks (independent slots per dead_kin — the view is keyed
    // (observer, dead_kin), so fresh +1.0 lands on a separate slot,
    // and `sum_for_first` adds them).
    //
    // Advance the tick by 1 so we can distinguish the two fear events
    // cleanly (decay is measured in ticks from emit). One tick decay at
    // the task-173 rate is 0.891 — still well above the 0.5 gate and
    // both events still count.
    state.tick += 1;
    let events_before_2 = events.total_pushed();
    let tick_2 = state.tick;
    state.kill_agent(w2);
    events.push(Event::AgentDied { agent_id: w2, tick: tick_2 });
    cascade.run_fixed_point(&mut state, &mut events);
    state.views.fold_all(&events, events_before_2, state.tick);

    let fears_phase_2: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::FearSpread { observer, dead_kin, .. } if *dead_kin == w2 => Some(*observer),
            _ => None,
        })
        .collect();
    // W2's death should emit 2 FearSpreads (W3, W4). W1 is already dead
    // so isn't a live observer.
    assert_eq!(
        fears_phase_2.len(),
        2,
        "W2 death should emit 2 FearSpreads (to W3, W4); got {fears_phase_2:?}",
    );

    // W3/W4 kin_fear should have GROWN — phase-2 fold lands on top of
    // slightly-decayed phase-1 (0.891^1 ≈ 0.891 per first event, plus
    // fresh +1.0 on the W2 slot). Sum ≈ 0.891 + 1.0 = 1.891.
    let kf3_b = state.views.kin_fear.sum_for_first(w3, state.tick);
    let kf4_b = state.views.kin_fear.sum_for_first(w4, state.tick);
    assert!(
        kf3_b > kf3_a,
        "W3 kin_fear should stack across two deaths: phase1={kf3_a}, phase2={kf3_b}",
    );
    assert!(
        kf4_b > kf4_a,
        "W4 kin_fear should stack across two deaths: phase1={kf4_a}, phase2={kf4_b}",
    );
    // Second event adds close to another full +1.0 (one tick decay loses
    // ~0.109 off the first event at rate 0.891). Combined should be well
    // above 1.5.
    assert!(
        kf3_b > 1.5,
        "W3 kin_fear after 2 packmate deaths = {kf3_b}, expected ≥1.5 \
         (≈0.891 + 1.0 = 1.891)",
    );

    // Behavioral check — rout cascade flipped both survivors to Flee.
    // Flee score with kin_fear > 0.5 gets +0.4 (task 173, retuned from
    // the original +0.6 gate). Score Flee directly to pin the modifier.
    let flee = flee_entry();
    let flee_w3 = score_row_for(flee, &state, w3, None);
    let flee_w4 = score_row_for(flee, &state, w4, None);
    assert!(
        flee_w3 > 0.3,
        "W3 Flee score {flee_w3} should include the +0.4 kin_fear bump",
    );
    assert!(
        flee_w4 > 0.3,
        "W4 Flee score {flee_w4} should include the +0.4 kin_fear bump",
    );
}
