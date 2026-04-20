// crates/engine/src/step.rs
use crate::cascade::CascadeRegistry;
use crate::channel::channel_range;
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::invariant::{FailureMode, InvariantRegistry};
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, ActionKind, AnnounceAudience, MacroAction, MicroTarget, PolicyBackend};
use crate::rng::per_agent_u32;
use crate::state::SimState;
use crate::telemetry::{metrics, NullSink, TelemetrySink};
use crate::view::MaterializedView;
use glam::Vec3;

/// Default vocal-strength multiplier used by `channel_range` when computing
/// speech-based announce radii. Real vocal-strength is a per-agent
/// Capability; until that lands, everyone shouts at 1.0. Audit fix MEDIUM #9.
pub const DEFAULT_VOCAL_STRENGTH: f32 = 1.0;

/// Return `true` iff `speaker` and `observer` share at least one
/// `CommunicationChannel`. When both sides have non-empty channel sets, at
/// least one must overlap. When either side has no registered channels
/// (e.g. cold-storage default, or a test state that never called
/// `spawn_agent`), we fall back to permissive so pre-channel-gating
/// fixtures don't silently go mute.
fn speaker_and_observer_share_channel(
    state: &SimState,
    speaker: AgentId,
    observer: AgentId,
) -> bool {
    let speaker_ch = state.agent_channels(speaker);
    let observer_ch = state.agent_channels(observer);
    match (speaker_ch, observer_ch) {
        (Some(s), Some(o)) => {
            if s.is_empty() || o.is_empty() { return true; }
            for c in s.iter() {
                if o.contains(c) { return true; }
            }
            false
        }
        _ => true,
    }
}

/// Longest effective range the `speaker` can project over any of their
/// registered channels, at `DEFAULT_VOCAL_STRENGTH`. Bounded above by
/// `MAX_ANNOUNCE_RADIUS` so a Telepathy-capable speaker doesn't broadcast
/// planet-wide. When the speaker has no registered channels, fall back to
/// `MAX_ANNOUNCE_RADIUS` (pre-channel-gating behaviour).
///
/// Only consulted for `AnnounceAudience::Anyone` / `Group(_)`; `Area(c, r)`
/// still uses the caller-supplied `r` (author intent).
fn speaker_anyone_radius(state: &SimState, speaker: AgentId) -> f32 {
    let channels = match state.agent_channels(speaker) {
        Some(c) if !c.is_empty() => c,
        _ => return MAX_ANNOUNCE_RADIUS,
    };
    let mut best: f32 = 0.0;
    for c in channels.iter() {
        let r = channel_range(*c, DEFAULT_VOCAL_STRENGTH);
        if r.is_infinite() {
            return MAX_ANNOUNCE_RADIUS;
        }
        if r.is_finite() && r > best {
            best = r;
        }
    }
    best.min(MAX_ANNOUNCE_RADIUS)
}

pub const MOVE_SPEED_MPS: f32 = 1.0;
pub const ATTACK_DAMAGE:  f32 = 10.0;
pub const ATTACK_RANGE:   f32 = 2.0;
const EAT_RESTORE:    f32 = 0.25;
const DRINK_RESTORE:  f32 = 0.30;
const REST_RESTORE:   f32 = 0.15;

/// Maximum number of `RecordMemory` events a single `Announce` can emit.
/// Bounds event-ring pressure even when the audience is very large.
pub const MAX_ANNOUNCE_RECIPIENTS: usize = 32;
/// Default hearing radius when `AnnounceAudience::Anyone` is used (and, for
/// the MVP, the fallback for `AnnounceAudience::Group(_)` until Task 16 wires
/// real group membership).
pub const MAX_ANNOUNCE_RADIUS:     f32  = 80.0;
/// Radius around the speaker within which non-recipient bystanders overhear an
/// `Announce` and record a lower-confidence memory (0.6 vs 0.8 for primary
/// recipients). Separate from `MAX_ANNOUNCE_RADIUS` — overhear always scans
/// around the speaker's position, regardless of the primary audience geometry.
pub const OVERHEAR_RANGE:          f32  = 30.0;

/// Apply a need-restoration desired delta to `current`, clamping the resulting
/// value at 1.0. Returns `(new_value, applied_delta)` where `applied_delta` is
/// the post-clamp change (≤ desired when saturated). Events carry the applied
/// delta so replays observe post-clamp values.
fn restore_need(current: f32, desired_delta: f32) -> (f32, f32) {
    let new_val = (current + desired_delta).min(1.0);
    let applied = new_val - current;
    (new_val, applied)
}

/// Convert the active effect-slow factor (q8 fixed-point) into an f32
/// multiplier. Returns 1.0 when no slow is active (`remaining == 0` OR
/// `factor_q8 <= 0`). The caller composes this multiplicatively with any
/// other speed modifiers (e.g. engagement-slow in the MoveToward branch).
///
/// q8 encoding: `factor_q8 = round(multiplier * 256)`. A factor of `51`
/// corresponds to ≈0.2× (51 / 256 ≈ 0.199).
fn effect_slow_multiplier(state: &SimState, id: AgentId) -> f32 {
    let remaining = state.agent_slow_remaining(id).unwrap_or(0);
    if remaining == 0 { return 1.0; }
    let factor_q8 = state.agent_slow_factor_q8(id).unwrap_or(0);
    if factor_q8 <= 0 { return 1.0; }
    factor_q8 as f32 / 256.0
}

/// Per-tick scratch buffers hoisted out of `step` so a steady-state tick loop
/// allocates zero bytes. Caller constructs once (capacity = `state.agent_cap()`),
/// reuses across ticks. Buffers are reset/cleared at the top of each `step`.
pub struct SimScratch {
    pub mask:        MaskBuffer,
    pub actions:     Vec<Action>,
    pub shuffle_idx: Vec<u32>,
}

impl SimScratch {
    pub fn new(n_agents: usize) -> Self {
        Self {
            mask:        MaskBuffer::new(n_agents),
            actions:     Vec::with_capacity(n_agents),
            shuffle_idx: Vec::with_capacity(n_agents),
        }
    }
}

/// Back-compat wrapper around [`step_full`]. Runs the canonical 6-phase pipeline
/// with an empty view list, an empty invariant registry, and a [`NullSink`] —
/// i.e. exactly the old Task 10 behavior (mask → evaluate → shuffle → apply →
/// cascade → tick++), with no view folds, no invariant checks, and no telemetry
/// emitted.
pub fn step<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing,
    backend: &B,
    cascade: &CascadeRegistry,
) {
    let empty_invariants = InvariantRegistry::new();
    step_full(
        state,
        scratch,
        events,
        backend,
        cascade,
        &mut [],
        &empty_invariants,
        &NullSink,
    );
}

/// Full 6-phase tick pipeline (see `docs/engine/spec.md` §12):
///
/// 1. Mask build
/// 2. Policy evaluate
/// 3. Action shuffle (deterministic per-tick Fisher-Yates)
/// 4. Apply actions + cascade fixed-point
/// 5. Materialized-view fold
/// 6. Invariants + built-in telemetry metrics
///
/// After phase 6, `state.tick` is incremented.
// The 8-param shape is load-bearing: it mirrors the Plan-2 canonical pipeline
// signature and the six observable phases each call out a distinct collaborator
// (state, scratch, events, backend, cascade, views, invariants, telemetry).
// Bundling would hide the phase seams from callers and tests.
#[allow(clippy::too_many_arguments)]
#[contracts::debug_requires(
    scratch.mask.micro_kind.len() == state.agent_cap() as usize * crate::mask::MicroKind::ALL.len()
)]
#[contracts::debug_ensures(state.tick == old(state.tick) + 1)]
pub fn step_full<B: PolicyBackend>(
    state:      &mut SimState,
    scratch:    &mut SimScratch,
    events:     &mut EventRing,
    backend:    &B,
    cascade:    &CascadeRegistry,
    views:      &mut [&mut dyn MaterializedView],
    invariants: &InvariantRegistry,
    telemetry:  &dyn TelemetrySink,
) {
    let t_start = std::time::Instant::now();

    // Combat Foundation Task 3 — unified tick-start phase. Runs before the
    // mask so mask predicates observe post-decrement, post-engagement state.
    // Emits StunExpired / SlowExpired on timer transitions to zero; updates
    // hot_engaged_with via bidirectional tentative-commit.
    crate::ability::expire::tick_start(state, events);

    // Phase 1 — mask build.
    scratch.mask.reset();
    scratch.mask.mark_hold_allowed(state);
    scratch.mask.mark_move_allowed_if_others_exist(state);
    scratch.mask.mark_flee_allowed_if_threat_exists(state);
    scratch.mask.mark_attack_allowed_if_target_in_range(state);
    scratch.mask.mark_needs_allowed(state);
    scratch.mask.mark_domain_hook_micros_allowed(
        state,
        cascade.cast_ability_registry().map(|arc| arc.as_ref()),
    );

    // Phase 2 — policy evaluate.
    scratch.actions.clear();
    backend.evaluate(state, &scratch.mask, &mut scratch.actions);

    // Phase 3 — deterministic per-tick action shuffle. Populates
    // `scratch.shuffle_idx` with a permutation over `scratch.actions` keyed by
    // `(state.seed, state.tick)`. `scratch.actions` itself is left untouched;
    // the apply kernel walks it via `shuffle_idx`.
    shuffle_actions_in_place(
        state.seed,
        state.tick,
        &scratch.actions,
        &mut scratch.shuffle_idx,
    );

    // Phase 4 — apply actions + run cascade fixed-point. Record events emitted
    // so phase 6 can report an accurate per-tick counter.
    let events_before = events.total_pushed();
    apply_actions(state, scratch, events);
    cascade.run_fixed_point_tel(state, events, telemetry);
    let events_emitted = events.total_pushed().saturating_sub(events_before);

    // Phase 5 — view fold. Each view walks the (retained subset of the) event
    // ring and accumulates its own derived storage.
    for v in views.iter_mut() {
        v.fold(events);
    }

    // Phase 6 — invariants + built-in telemetry metrics.
    let violations = invariants.check_all(state, events);
    for report in &violations {
        let mode_str = match report.failure_mode {
            FailureMode::Panic => "panic",
            FailureMode::Log   => "log",
        };
        telemetry.emit(
            "engine.invariant_violated",
            1.0,
            &[("invariant", report.violation.invariant), ("mode", mode_str)],
        );
    }

    let tick_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    telemetry.emit_histogram(metrics::TICK_MS, tick_ms);
    telemetry.emit_counter(metrics::EVENT_COUNT, events_emitted as i64);
    let n_alive = state.agents_alive().count();
    telemetry.emit(metrics::AGENT_ALIVE, n_alive as f64, &[]);
    let mask_true_frac = fraction_true(&scratch.mask.micro_kind);
    telemetry.emit(metrics::MASK_TRUE_FRAC, mask_true_frac, &[]);

    state.tick += 1;
}

fn fraction_true(bits: &[bool]) -> f64 {
    if bits.is_empty() { return 0.0; }
    let t = bits.iter().filter(|b| **b).count();
    t as f64 / bits.len() as f64
}

/// Phase-3 helper. Populates `shuffle_idx` with a Fisher-Yates permutation of
/// `0..actions.len()`, keyed by `(world_seed, tick)` via [`per_agent_u32`]
/// using the sentinel `AgentId(1)` as the shuffle stream discriminator.
/// Deterministic: same `(seed, tick, actions.len())` → same permutation.
fn shuffle_actions_in_place(
    world_seed:  u64,
    tick:        u32,
    actions:     &[Action],
    shuffle_idx: &mut Vec<u32>,
) {
    shuffle_order_into(shuffle_idx, actions.len(), world_seed, tick);
}

/// Fisher-Yates shuffle of action indices using a deterministic PRNG seeded by
/// `(world_seed, tick)`. This makes action-application order depend on the world
/// seed (spec §7.2 — determinism contract / first-mover-bias prevention).
///
/// Writes into the caller-owned `order` buffer (cleared + extended in place) so
/// the per-tick order vec does not re-allocate once `SimScratch` is warm.
fn shuffle_order_into(order: &mut Vec<u32>, n: usize, world_seed: u64, tick: u32) {
    order.clear();
    order.extend(0..n as u32);
    let tick64 = tick as u64;
    // Sentinel agent id 1 is used as a fixed stream discriminator for the
    // per-tick shuffle — distinct from any per-agent decision stream.
    let sentinel = AgentId::new(1).unwrap();
    for i in (1..n).rev() {
        let r = per_agent_u32(world_seed, sentinel, tick64 * 65536 + i as u64, b"shuffle");
        let j = (r as usize) % (i + 1);
        order.swap(i, j);
    }
}

fn apply_actions(
    state:   &mut SimState,
    scratch: &SimScratch,
    events:  &mut EventRing,
) {
    // `scratch.shuffle_idx` must have been populated by `shuffle_actions_in_place`
    // immediately before this call. We walk the already-computed permutation
    // rather than re-shuffling here, so the shuffle is a first-class phase of
    // `step_full` visible to telemetry / tests.
    for &idx in scratch.shuffle_idx.iter() {
        let action = &scratch.actions[idx as usize];
        match action.kind {
            ActionKind::Micro { kind: MicroKind::Hold, .. } => {}
            ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(target_pos),
            } => {
                let from = state.agent_pos(action.agent).unwrap();
                let delta = target_pos - from;
                if delta.length_squared() > 0.0 {
                    let dir = delta.normalize();
                    // Combat Foundation Task 4 — engaged-aware movement.
                    // Moving *toward* the engager is full speed (closing the
                    // melee); moving anywhere else is slowed by
                    // ENGAGEMENT_SLOW_FACTOR and fires an opportunity attack.
                    let mut speed = MOVE_SPEED_MPS;
                    if let Some(engager) = state.agent_engaged_with(action.agent) {
                        let engager_pos = state.agent_pos(engager).unwrap_or(from);
                        let toward_engager = (engager_pos - from).dot(dir) > 0.0;
                        if !toward_engager {
                            speed *= crate::ability::expire::ENGAGEMENT_SLOW_FACTOR;
                            events.push(Event::OpportunityAttackTriggered {
                                attacker: engager,
                                target:   action.agent,
                                tick:     state.tick,
                            });
                        }
                    }
                    // Combat Foundation Task 14 — effect-slow multiplier.
                    // `hot_slow_factor_q8` is q8 fixed-point (256 = 1.0×). A
                    // remaining-ticks > 0 means a slow debuff is active;
                    // compose MULTIPLICATIVELY with engagement-slow so both
                    // sources stack predictably.
                    speed *= effect_slow_multiplier(state, action.agent);
                    let to = from + dir * speed;
                    state.set_agent_pos(action.agent, to);
                    events.push(Event::AgentMoved {
                        agent_id: action.agent, from, to, tick: state.tick,
                    });
                }
            }
            ActionKind::Micro {
                kind:   MicroKind::Flee,
                target: MicroTarget::Agent(threat),
            } => {
                if !state.agent_alive(threat) { continue; }
                if let (Some(self_pos), Some(threat_pos)) =
                    (state.agent_pos(action.agent), state.agent_pos(threat))
                {
                    let away = (self_pos - threat_pos).normalize_or_zero();
                    if away.length_squared() > 0.0 {
                        // Flee intentionally disengages at full speed but
                        // always draws an opportunity attack from any active
                        // engager (regardless of whether the engager is the
                        // threat or someone else).
                        if let Some(engager) = state.agent_engaged_with(action.agent) {
                            events.push(Event::OpportunityAttackTriggered {
                                attacker: engager,
                                target:   action.agent,
                                tick:     state.tick,
                            });
                        }
                        // Effect-slow applies to Flee too (Task 14) —
                        // composed multiplicatively with the Flee base speed.
                        let speed = MOVE_SPEED_MPS * effect_slow_multiplier(state, action.agent);
                        let new_pos = self_pos + away * speed;
                        state.set_agent_pos(action.agent, new_pos);
                        events.push(Event::AgentFled {
                            agent_id: action.agent,
                            from:     self_pos,
                            to:       new_pos,
                            tick:     state.tick,
                        });
                    }
                }
            }
            ActionKind::Micro {
                kind:   MicroKind::Attack,
                target: MicroTarget::Agent(tgt),
            } => {
                if !state.agent_alive(tgt) { continue; }
                if let (Some(sp), Some(tp)) =
                    (state.agent_pos(action.agent), state.agent_pos(tgt))
                {
                    // Audit fix MEDIUM #10: honour per-agent attack stats.
                    // Default falls back to the module constants so legacy
                    // call sites that never touched the setters behave
                    // identically to the pre-port kernel.
                    let range = state.agent_attack_range(action.agent).unwrap_or(ATTACK_RANGE);
                    let damage = state.agent_attack_damage(action.agent).unwrap_or(ATTACK_DAMAGE);
                    if sp.distance(tp) <= range {
                        let new_hp = (state.agent_hp(tgt).unwrap_or(0.0) - damage).max(0.0);
                        state.set_agent_hp(tgt, new_hp);
                        events.push(Event::AgentAttacked {
                            attacker: action.agent,
                            target:   tgt,
                            damage,
                            tick:     state.tick,
                        });
                        if new_hp <= 0.0 {
                            events.push(Event::AgentDied {
                                agent_id: tgt,
                                tick:     state.tick,
                            });
                            state.kill_agent(tgt);
                        }
                    }
                }
            }
            ActionKind::Micro { kind: MicroKind::Eat, .. } => {
                if let Some(cur) = state.agent_hunger(action.agent) {
                    let (new_val, applied) = restore_need(cur, EAT_RESTORE);
                    state.set_agent_hunger(action.agent, new_val);
                    events.push(Event::AgentAte {
                        agent_id: action.agent, delta: applied, tick: state.tick,
                    });
                }
            }
            ActionKind::Micro { kind: MicroKind::Drink, .. } => {
                if let Some(cur) = state.agent_thirst(action.agent) {
                    let (new_val, applied) = restore_need(cur, DRINK_RESTORE);
                    state.set_agent_thirst(action.agent, new_val);
                    events.push(Event::AgentDrank {
                        agent_id: action.agent, delta: applied, tick: state.tick,
                    });
                }
            }
            ActionKind::Micro { kind: MicroKind::Rest, .. } => {
                if let Some(cur) = state.agent_rest_timer(action.agent) {
                    let (new_val, applied) = restore_need(cur, REST_RESTORE);
                    state.set_agent_rest_timer(action.agent, new_val);
                    events.push(Event::AgentRested {
                        agent_id: action.agent, delta: applied, tick: state.tick,
                    });
                }
            }
            // Combat Foundation Task 9 — Cast dispatch. Push one `AgentCast`
            // event; the `CastHandler` cascade looks the program up in its
            // `AbilityRegistry` and emits one `Effect*Applied` per op.
            //
            // Root casts start at `depth = 0`; the CastHandler increments
            // for each nested `EffectOp::CastAbility` emission (Task 18).
            ActionKind::Micro {
                kind: MicroKind::Cast,
                target: MicroTarget::Ability { id, target },
            } => {
                events.push(Event::AgentCast {
                    caster:  action.agent,
                    ability: id,
                    target,
                    depth:   0,
                    tick:    state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::UseItem,
                target: MicroTarget::ItemSlot(slot),
            } => {
                events.push(Event::AgentUsedItem {
                    agent_id: action.agent, item_slot: slot, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::Harvest,
                target: MicroTarget::Opaque(r),
            } => {
                events.push(Event::AgentHarvested {
                    agent_id: action.agent, resource: r, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::PlaceTile,
                target: MicroTarget::Position(p),
            } => {
                events.push(Event::AgentPlacedTile {
                    agent_id: action.agent, where_pos: p, kind_tag: 0, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::PlaceVoxel,
                target: MicroTarget::Position(p),
            } => {
                events.push(Event::AgentPlacedVoxel {
                    agent_id: action.agent, where_pos: p, mat_tag: 0, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::HarvestVoxel,
                target: MicroTarget::Position(p),
            } => {
                events.push(Event::AgentHarvestedVoxel {
                    agent_id: action.agent, where_pos: p, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::Converse,
                target: MicroTarget::Agent(b),
            } => {
                events.push(Event::AgentConversed {
                    agent_id: action.agent, partner: b, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::ShareStory,
                target: MicroTarget::Opaque(topic),
            } => {
                events.push(Event::AgentSharedStory {
                    agent_id: action.agent, topic, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::Communicate,
                target: MicroTarget::Agent(r),
            } => {
                events.push(Event::AgentCommunicated {
                    speaker: action.agent, recipient: r, fact_ref: 0, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::Ask,
                target: MicroTarget::Agent(t),
            } => {
                events.push(Event::InformationRequested {
                    asker: action.agent, target: t, query: 0, tick: state.tick,
                });
            }
            ActionKind::Micro {
                kind: MicroKind::Remember,
                target: MicroTarget::Opaque(s),
            } => {
                events.push(Event::AgentRemembered {
                    agent_id: action.agent, subject: s, tick: state.tick,
                });
            }
            ActionKind::Micro { .. } => {
                // Ill-formed actions (kind + target-type mismatch) are silently
                // dropped. Mask predicates should prevent well-behaved backends
                // from landing here.
            }
            ActionKind::Macro(MacroAction::NoOp) => { /* nothing */ }
            ActionKind::Macro(MacroAction::PostQuest { quest_id, category, resolution }) => {
                events.push(Event::QuestPosted {
                    poster: action.agent,
                    quest_id,
                    category,
                    resolution,
                    tick: state.tick,
                });
            }
            ActionKind::Macro(MacroAction::AcceptQuest { quest_id, acceptor }) => {
                events.push(Event::QuestAccepted {
                    acceptor,
                    quest_id,
                    tick: state.tick,
                });
            }
            ActionKind::Macro(MacroAction::Bid { auction_id, bidder, amount }) => {
                events.push(Event::BidPlaced {
                    bidder,
                    auction_id,
                    amount,
                    tick: state.tick,
                });
            }
            ActionKind::Macro(MacroAction::Announce { speaker, audience, fact_payload }) => {
                events.push(Event::AnnounceEmitted {
                    speaker,
                    audience_tag: audience.tag(),
                    fact_payload,
                    tick: state.tick,
                });
                // Channel-gated radius for `Anyone` / `Group` audiences:
                // speaker's longest-reach channel at default vocal strength,
                // bounded by MAX_ANNOUNCE_RADIUS. `Area(c, r)` still uses the
                // caller-supplied `r` (author intent). Audit fix MEDIUM #9.
                let anyone_radius = speaker_anyone_radius(state, speaker);
                let (center, radius) = match audience {
                    AnnounceAudience::Area(c, r) => (c, r),
                    AnnounceAudience::Anyone => {
                        let sp = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                        (sp, anyone_radius)
                    }
                    AnnounceAudience::Group(_) => {
                        // TODO(Task 16): use group membership. For MVP, fall back to Anyone.
                        let sp = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                        (sp, anyone_radius)
                    }
                };
                // Deterministic iteration: enumerate spatial-index hits in
                // slot order (agents_alive() is slot-order) so the first
                // MAX_ANNOUNCE_RECIPIENTS within range is reproducible.
                // Audit fix CRITICAL #1: consume the spatial index so audience
                // enumeration is sub-linear; we walk `agents_alive()` in slot
                // order and test membership in the candidate set to preserve
                // the deterministic first-MAX_ANNOUNCE_RECIPIENTS semantics.
                let spatial = state.spatial();
                let candidates: smallvec::SmallVec<[AgentId; 64]> = spatial
                    .query_within_radius(state, center, radius)
                    .collect();
                let mut primary_observers: smallvec::SmallVec<[AgentId; 32]> =
                    smallvec::SmallVec::new();
                let mut count = 0usize;
                for obs in state.agents_alive() {
                    if count >= MAX_ANNOUNCE_RECIPIENTS { break; }
                    if obs == speaker { continue; }
                    if !candidates.contains(&obs) { continue; }
                    // Audit fix MEDIUM #9: channel-eligibility filter.
                    if !speaker_and_observer_share_channel(state, speaker, obs) { continue; }
                    events.push(Event::RecordMemory {
                        observer:     obs,
                        source:       speaker,
                        fact_payload,
                        confidence:   0.8,
                        tick:         state.tick,
                    });
                    primary_observers.push(obs);
                    count += 1;
                }

                // Overhear scan: bystanders within OVERHEAR_RANGE of the
                // SPEAKER (not the audience center) who were not primary
                // recipients get a lower-confidence memory. Speaker excluded.
                let speaker_pos = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                let overhear_candidates: smallvec::SmallVec<[AgentId; 64]> = spatial
                    .query_within_radius(state, speaker_pos, OVERHEAR_RANGE)
                    .collect();
                for obs in state.agents_alive() {
                    if obs == speaker { continue; }
                    if primary_observers.contains(&obs) { continue; }
                    if !overhear_candidates.contains(&obs) { continue; }
                    if !speaker_and_observer_share_channel(state, speaker, obs) { continue; }
                    events.push(Event::RecordMemory {
                        observer:     obs,
                        source:       speaker,
                        fact_payload,
                        confidence:   0.6,
                        tick:         state.tick,
                    });
                }
            }
        }
    }
}

