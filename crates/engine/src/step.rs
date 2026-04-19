// crates/engine/src/step.rs
use crate::cascade::CascadeRegistry;
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::{Action, ActionKind, AnnounceAudience, MacroAction, MicroTarget, PolicyBackend};
use crate::rng::per_agent_u32;
use crate::state::SimState;
use glam::Vec3;

const MOVE_SPEED_MPS: f32 = 1.0;
const ATTACK_DAMAGE:  f32 = 10.0;
const ATTACK_RANGE:   f32 = 2.0;
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

pub fn step<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing,
    backend: &B,
    cascade: &CascadeRegistry,
) {
    scratch.mask.reset();
    scratch.mask.mark_hold_allowed(state);
    scratch.mask.mark_move_allowed_if_others_exist(state);
    scratch.mask.mark_flee_allowed_if_threat_exists(state);
    scratch.mask.mark_attack_allowed_if_target_in_range(state);
    scratch.mask.mark_needs_allowed(state);
    scratch.mask.mark_domain_hook_micros_allowed(state);
    scratch.actions.clear();
    backend.evaluate(state, &scratch.mask, &mut scratch.actions);

    apply_actions(state, &scratch.actions, events, &mut scratch.shuffle_idx);
    cascade.run_fixed_point(state, events);
    state.tick += 1;
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
    actions: &[Action],
    events:  &mut EventRing,
    order:   &mut Vec<u32>,
) {
    shuffle_order_into(order, actions.len(), state.seed, state.tick);
    for &idx in order.iter() {
        let action = &actions[idx as usize];
        match action.kind {
            ActionKind::Micro { kind: MicroKind::Hold, .. } => {}
            ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(target_pos),
            } => {
                let from = state.agent_pos(action.agent).unwrap();
                let delta = target_pos - from;
                if delta.length_squared() > 0.0 {
                    let to = from + delta.normalize() * MOVE_SPEED_MPS;
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
                        let new_pos = self_pos + away * MOVE_SPEED_MPS;
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
                    if sp.distance(tp) <= ATTACK_RANGE {
                        let new_hp = (state.agent_hp(tgt).unwrap_or(0.0) - ATTACK_DAMAGE).max(0.0);
                        state.set_agent_hp(tgt, new_hp);
                        events.push(Event::AgentAttacked {
                            attacker: action.agent,
                            target:   tgt,
                            damage:   ATTACK_DAMAGE,
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
            // Event-only micros — engine emits the typed event; domain-specific
            // effects land as compiler-registered cascade handlers in later plans.
            ActionKind::Micro {
                kind: MicroKind::Cast,
                target: MicroTarget::AbilityIdx(idx),
            } => {
                events.push(Event::AgentCast {
                    agent_id: action.agent, ability_idx: idx, tick: state.tick,
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
                let (center, radius) = match audience {
                    AnnounceAudience::Area(c, r) => (c, r),
                    AnnounceAudience::Anyone => {
                        let sp = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                        (sp, MAX_ANNOUNCE_RADIUS)
                    }
                    AnnounceAudience::Group(_) => {
                        // TODO(Task 16): use group membership. For MVP, fall back to Anyone.
                        let sp = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                        (sp, MAX_ANNOUNCE_RADIUS)
                    }
                };
                // Deterministic iteration: agents_alive() walks slots in order,
                // so the first MAX_ANNOUNCE_RECIPIENTS within range is reproducible.
                // SmallVec keeps the overhear-dedup check allocation-free for
                // steady-state ticks (dhat-heap test doesn't emit Announce, but
                // keeping this hot-path-clean matches the rest of `step`).
                let mut primary_observers: smallvec::SmallVec<[AgentId; 32]> =
                    smallvec::SmallVec::new();
                let mut count = 0usize;
                for obs in state.agents_alive() {
                    if count >= MAX_ANNOUNCE_RECIPIENTS { break; }
                    if obs == speaker { continue; }
                    let op = match state.agent_pos(obs) {
                        Some(p) => p,
                        None    => continue,
                    };
                    if op.distance(center) <= radius {
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
                }

                // Overhear scan: bystanders within OVERHEAR_RANGE of the
                // SPEAKER (not the audience center) who were not primary
                // recipients get a lower-confidence memory. Speaker excluded.
                let speaker_pos = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                for obs in state.agents_alive() {
                    if obs == speaker { continue; }
                    if primary_observers.contains(&obs) { continue; }
                    let op = match state.agent_pos(obs) {
                        Some(p) => p,
                        None    => continue,
                    };
                    if op.distance(speaker_pos) <= OVERHEAR_RANGE {
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
}

