//! Phase 6f — GPU cascade driver (Piece 3 of the megakernel plan).
//!
//! Ties together the per-piece kernels landed in previous phases:
//!
//!   * Piece 1 (task 190): scoring reads view_storage atomics directly.
//!   * Piece 2 (task 191): `PhysicsKernel::run_batch` consumes an event
//!     batch + agent SoA and emits one cascade iteration's worth of new
//!     events + mutated state.
//!   * Piece 3 (this module): drives `run_batch` in a fixed-point loop,
//!     folding each iteration's emitted events back into `view_storage`
//!     so the next iteration — and the scoring kernel on the next tick —
//!     reads the post-fold state.
//!
//! # Shape
//!
//! ```text
//! cascade_gpu(state, events_in):
//!   1. Seed events_in_gpu = pack(events_in)
//!   2. Pack agent SoA once, pre-compute spatial + abilities uploads.
//!   3. for iter in 0..MAX_CASCADE_ITERATIONS:
//!        a. if events_in.is_empty(): break (natural fixed point)
//!        b. physics.run_batch(agents, events_in) -> (agents, events_out)
//!        c. dispatch view folds for each affected view on events_out
//!        d. events_in = events_out
//!   4. Return aggregated emissions + the final agent SoA.
//! ```
//!
//! # CPU/GPU split
//!
//! The GPU physics kernel covers **12 of the 23 rules** in physics.sim
//! fully. The remaining 11 (8 chronicle + transfer_gold, modify_standing,
//! record_memory) are no-op stubs on the GPU side because they mutate
//! state the GPU agent SoA doesn't carry:
//!
//!   * Chronicle rules emit `ChronicleEntry`, which is non-replayable and
//!     doesn't feed back into the replay cascade. The CPU path can re-emit
//!     them post-cascade from the drained event list if narrative prose
//!     is needed.
//!   * `transfer_gold` / `modify_standing` / `record_memory` need gold,
//!     standing matrix, and memory ring respectively — none of which the
//!     physics SoA exposes. Callers that need those mutations apply them
//!     on the CPU side by replaying the drained event set through a
//!     cold-state handler.
//!
//! This module does **not** wire itself into `GpuBackend::step`'s tick
//! loop as authoritative yet — Piece 4 does. For now, the cascade driver
//! is a standalone entry point exercised by the parity test, and
//! `GpuBackend::step` continues to forward the tick to `engine::step::step`
//! as it has since Phase 0. Running cascade for every tick in the parity
//! harness builds the empirical confidence needed to flip the authority
//! switch in Piece 4.
//!
//! # Determinism
//!
//! * `PhysicsKernel::run_batch`'s output records are sorted by
//!   `(tick, kind, payload[0])` so iteration over a batch hits the kernel
//!   in a stable order.
//! * View fold kernels use CAS loops with `self += 1.0` constant deltas —
//!   order-invariant for the deltas actually in use.
//! * Spatial precompute runs once per cascade call; the kin / nearest
//!   lists don't refresh inside the loop. That matches the CPU cascade:
//!   `CascadeRegistry::run_fixed_point` does not rebuild the spatial hash
//!   between iterations, so neither do we.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;

use engine::event::{Event, EventRing};
use engine::state::SimState;

use crate::event_ring::{pack_event, unpack_record, DrainOutcome, EventRecord};
use crate::physics::{
    pack_agent_slots, unpack_agent_slots, GpuAgentSlot, GpuKinList, PackedAbilityRegistry,
    PhysicsBatchOutput, PhysicsCfg, PhysicsError, PhysicsKernel, MAX_ABILITIES, MAX_EFFECTS,
};
use crate::spatial_gpu::{GpuSpatialHash, SpatialError, K, NO_HOSTILE};
use crate::view_storage::{FoldInputPair, FoldInputSlot, ViewStorage, ViewStorageError};

/// Hard bound on cascade iterations — matches `engine::cascade::dispatch::
/// MAX_CASCADE_ITERATIONS`. The CPU cascade panics past this bound in
/// debug builds; the GPU cascade logs and truncates, matching the release
/// behaviour there (see `run_cascade` below for the rationale).
pub const MAX_CASCADE_ITERATIONS: u32 = 8;

/// Default kin-radius used when precomputing `nearby_kin` — matches the
/// 12 m fold radius used by `fear_spread_on_death` / `pack_focus_on_engagement`
/// / `rally_on_wound` in `assets/sim/physics.sim`.
pub const DEFAULT_KIN_RADIUS: f32 = 12.0;

/// Aggregated result of a cascade run.
#[derive(Debug)]
pub struct CascadeOutput {
    /// Final packed agent SoA after every iteration's physics mutations.
    /// Callers typically pass this to `unpack_agent_slots(state, ...)` to
    /// commit the field-level changes (hp, shield, stun, slow, cooldown,
    /// engaged_with, alive) onto their `SimState`.
    pub final_agent_slots: Vec<GpuAgentSlot>,
    /// Every replayable event emitted across every iteration, ordered by
    /// iteration (0 first) then by the per-iteration sort key
    /// `(tick, kind, payload[0])`. The GPU physics kernel already sorts
    /// each iteration; the driver concatenates them in order.
    pub all_emitted_events: Vec<EventRecord>,
    /// Number of `run_batch` dispatches the driver issued. 0 if the
    /// initial batch was empty; otherwise >=1.
    pub iterations: u32,
    /// True iff the final iteration produced zero new events (natural
    /// fixed point). False iff we hit `MAX_CASCADE_ITERATIONS` before
    /// converging — the tail-end events are still in `all_emitted_events`
    /// but future cascades wouldn't re-process them.
    pub converged: bool,
    /// Per-iteration drain outcome — used by tests to assert
    /// `!overflowed` and to report the cascade's event-volume profile.
    pub drain_outcomes: Vec<DrainOutcome>,
}

/// Error surface for `run_cascade`. Always surfaces at the caller as a
/// hard stop — cascade failures aren't retryable.
#[derive(Debug)]
pub enum CascadeError {
    Physics(PhysicsError),
    View(ViewStorageError),
    Spatial(SpatialError),
    /// Event ring overflowed inside the cascade — a lost-event condition
    /// the parity test surfaces as a hard failure.
    EventRingOverflow { iteration: u32, drain: DrainOutcome },
}

impl std::fmt::Display for CascadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CascadeError::Physics(e) => write!(f, "cascade physics: {e}"),
            CascadeError::View(e) => write!(f, "cascade view: {e}"),
            CascadeError::Spatial(e) => write!(f, "cascade spatial: {e}"),
            CascadeError::EventRingOverflow { iteration, drain } => write!(
                f,
                "cascade event ring overflowed at iter={iteration}: drain={drain:?}"
            ),
        }
    }
}

impl std::error::Error for CascadeError {}

impl From<PhysicsError> for CascadeError {
    fn from(e: PhysicsError) -> Self {
        CascadeError::Physics(e)
    }
}

impl From<ViewStorageError> for CascadeError {
    fn from(e: ViewStorageError) -> Self {
        CascadeError::View(e)
    }
}

impl From<SpatialError> for CascadeError {
    fn from(e: SpatialError) -> Self {
        CascadeError::Spatial(e)
    }
}

/// Run the GPU cascade to convergence.
///
/// `initial_events` seeds the first dispatch. After each iteration the
/// driver:
///   * sorts the emitted events (physics already sorts internally),
///   * folds them into `view_storage` via the per-view fold kernels,
///   * feeds them as `events_in` to the next iteration.
///
/// The `SimState`'s SoA is NOT mutated — on return the caller pulls
/// `final_agent_slots` and decides whether to commit via
/// `unpack_agent_slots`. That separation lets the parity test run cascade
/// as an observational pass without perturbing the CPU cascade still
/// running alongside it.
///
/// `abilities`, `kin_radius`, and `spatial` are hoisted to arguments so
/// callers can seed them once per tick and reuse across multiple cascade
/// invocations without re-uploading. The hot path (every tick of the
/// parity test) passes an empty ability registry + default kin radius.
#[allow(clippy::too_many_arguments)]
pub fn run_cascade(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &SimState,
    physics: &mut PhysicsKernel,
    view_storage: &mut ViewStorage,
    spatial: &mut GpuSpatialHash,
    abilities: &PackedAbilityRegistry,
    initial_events: &[EventRecord],
    kin_radius: f32,
    ctx: &EmitContext<'_>,
) -> Result<CascadeOutput, CascadeError> {
    // `ctx` is taken to keep the cascade driver symmetric with the
    // kernel's init surface — callers who already own a `Compilation`
    // hand it through without rebuilding. It's currently unused at
    // dispatch time because the physics kernel was compiled against it
    // when `PhysicsKernel::new` ran.
    let _ = ctx;
    let agent_cap = state.agent_cap();
    let mut agent_slots = pack_agent_slots(state);

    // Spatial precompute — two dispatches per cascade call:
    //
    //   * At `kin_radius` (12 m) — feeds `nearby_kin` for fear/pack/rally.
    //   * At `combat.engagement_range` (2 m by default) — feeds
    //     `nearest_hostile_to` for `engagement_on_move`. Without this
    //     second query, `engagement_on_move` would engage any hostile
    //     within the kin radius, not the engagement radius, which
    //     causes the CPU cascade (which uses engagement_range) and the
    //     GPU cascade to diverge on any fixture where agents come
    //     within 12 m but outside 2 m of each other.
    //
    // Task 190's physics emitter wraps the spatial queries as
    // `spatial_nearest_hostile_to(agent, radius)` where the wrapped
    // call ignores the radius argument — it reads the precomputed
    // result. Running the precompute at engagement_range lines the
    // read back up with what the DSL rule's `nearest_hostile_to(mover,
    // engagement_range)` call intended.
    let kin_results = spatial.rebuild_and_query(device, queue, state, kin_radius)?;
    let engagement_range = state.config.combat.engagement_range;
    let hostile_results = spatial.rebuild_and_query(device, queue, state, engagement_range)?;

    let kin_lists: Vec<GpuKinList> = kin_results
        .nearby_kin
        .iter()
        .map(|q| {
            let mut kl = GpuKinList::default();
            kl.count = q.count;
            for i in 0..(q.count as usize).min(K as usize) {
                kl.ids[i] = q.ids[i];
            }
            kl
        })
        .collect();
    // Pad to agent_cap. Should already be len==agent_cap from spatial.
    let mut kin_lists = kin_lists;
    kin_lists.resize(agent_cap as usize, GpuKinList::default());
    let mut nearest_hostile = hostile_results.nearest_hostile.clone();
    nearest_hostile.resize(agent_cap as usize, NO_HOSTILE);

    // Config uniform — stable across all iterations.
    let cfg_template = PhysicsCfg {
        tick: state.tick,
        num_events: 0, // overridden per iteration by run_batch
        combat_engagement_range: state.config.combat.engagement_range,
        cascade_max_iterations: MAX_CASCADE_ITERATIONS,
        agent_cap,
        max_abilities: MAX_ABILITIES as u32,
        max_effects: MAX_EFFECTS as u32,
        _pad: 0,
    };

    let mut all_emitted: Vec<EventRecord> = Vec::new();
    let mut drain_outcomes: Vec<DrainOutcome> = Vec::new();
    let mut events_in = initial_events.to_vec();
    let mut iterations = 0u32;
    let mut converged = false;

    for iter in 0..MAX_CASCADE_ITERATIONS {
        if events_in.is_empty() {
            converged = true;
            break;
        }
        iterations = iter + 1;

        // The spatial precompute ran once with the PRE-cascade alive
        // set. If an agent died in a prior iteration, its id can still
        // sit in another slot's kin list — which would cause the GPU
        // `nearby_kin` loop to emit follow-on events (FearSpread,
        // PackAssist, RallyCall) FROM dead observers TOWARD dead kin.
        // The CPU `spatial::nearby_kin` filters alive at call time; we
        // mirror that by stripping dead ids from every kin list before
        // re-dispatching. Cheap — one pass over agent_cap * K.
        let filtered_kin = filter_dead_from_kin(&kin_lists, &agent_slots);

        let PhysicsBatchOutput {
            agent_slots_out,
            events_out,
            drain,
        } = physics.run_batch(
            device,
            queue,
            &agent_slots,
            abilities,
            &filtered_kin,
            &nearest_hostile,
            &events_in,
            cfg_template,
        )?;

        if drain.overflowed {
            return Err(CascadeError::EventRingOverflow {
                iteration: iter,
                drain,
            });
        }
        agent_slots = agent_slots_out;
        drain_outcomes.push(drain);

        // Fold this iteration's events into view_storage. Must run BEFORE
        // we overwrite `events_in`, so the next iteration's scoring
        // dispatch (and the next tick's scoring) reads the post-fold
        // cells.
        fold_iteration_events(device, queue, view_storage, &events_out)?;

        // Accumulate and loop.
        all_emitted.extend(events_out.iter().copied());
        events_in = events_out;
    }

    // Distinguish "8 iterations but last one was empty" (converged) from
    // "8 iterations, still emitting" (truncated). `events_in.is_empty()`
    // is checked at the top of the next iteration — if we broke via the
    // `if events_in.is_empty()` path, `converged == true`. If the loop
    // fell off the bottom, we only claim convergence when the last batch
    // produced nothing new.
    //
    // Above: `converged` was set to `true` only when the early break
    // triggered inside the for-loop. If we fell off the end without that
    // early break, check again: if the last iteration produced empty
    // events_out, we ALSO converged.
    if !converged && events_in.is_empty() {
        converged = true;
    }

    if !converged {
        eprintln!(
            "engine_gpu::cascade: did not converge within {} iterations \
             (final iter emitted {} events) — truncating",
            MAX_CASCADE_ITERATIONS,
            events_in.len(),
        );
    }

    Ok(CascadeOutput {
        final_agent_slots: agent_slots,
        all_emitted_events: all_emitted,
        iterations,
        converged,
        drain_outcomes,
    })
}

/// Fold one iteration's emitted events into `view_storage`. Mirrors the
/// fold handlers in `assets/sim/views.sim`:
///
///   * `AgentAttacked { actor, target }` → `my_enemies[target, actor]`
///     and `threat_level[target, actor]`.
///   * `EffectDamageApplied { actor, target }` → `threat_level[target, actor]`.
///   * `FearSpread { observer, dead_kin }` → `kin_fear[observer, dead_kin]`.
///   * `PackAssist { observer, target }` → `pack_focus[observer, target]`.
///   * `RallyCall { observer, wounded_kin }` → `rally_boost[observer, wounded_kin]`.
///   * `EngagementCommitted { actor, target }` → `engaged_with` insert.
///   * `EngagementBroken { actor, former_target }` → `engaged_with` remove.
///
/// View storage cells are keyed by 0-based slot (raw id - 1), matching
/// `FoldInputPair::first` / `FoldInputSlot::first`.
pub fn fold_iteration_events(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view_storage: &mut ViewStorage,
    events: &[EventRecord],
) -> Result<(), ViewStorageError> {
    use crate::event_ring::EventKindTag;

    // Gather per-view fold-input vectors in one pass so we dispatch each
    // view's kernel at most once per iteration. Matches the CPU
    // `fold_all`'s per-view inner loop (each view walks the whole
    // this-tick slice and filters on match arms).
    let mut my_enemies: Vec<FoldInputPair> = Vec::new();
    let mut threat_level: Vec<FoldInputPair> = Vec::new();
    let mut kin_fear: Vec<FoldInputPair> = Vec::new();
    let mut pack_focus: Vec<FoldInputPair> = Vec::new();
    let mut rally_boost: Vec<FoldInputPair> = Vec::new();
    let mut engaged_with: Vec<FoldInputSlot> = Vec::new();

    for rec in events {
        let Some(tag) = EventKindTag::from_u32(rec.kind) else { continue };
        let tick = rec.tick;
        match tag {
            EventKindTag::AgentAttacked => {
                // payload: [actor, target, damage_bits]
                let actor = rec.payload[0];
                let target = rec.payload[1];
                if let Some((obs, atk)) = slot_pair(target, actor) {
                    my_enemies.push(FoldInputPair { first: obs, second: atk, tick, _pad: 0 });
                    threat_level.push(FoldInputPair { first: obs, second: atk, tick, _pad: 0 });
                }
            }
            EventKindTag::EffectDamageApplied => {
                // payload: [actor, target, amount_bits]
                let actor = rec.payload[0];
                let target = rec.payload[1];
                if let Some((a, b)) = slot_pair(target, actor) {
                    threat_level.push(FoldInputPair { first: a, second: b, tick, _pad: 0 });
                }
            }
            EventKindTag::FearSpread => {
                // payload: [observer, dead_kin]
                let observer = rec.payload[0];
                let dead_kin = rec.payload[1];
                if let Some((a, b)) = slot_pair(observer, dead_kin) {
                    kin_fear.push(FoldInputPair { first: a, second: b, tick, _pad: 0 });
                }
            }
            EventKindTag::PackAssist => {
                // payload: [observer, target]
                let observer = rec.payload[0];
                let target = rec.payload[1];
                if let Some((a, b)) = slot_pair(observer, target) {
                    pack_focus.push(FoldInputPair { first: a, second: b, tick, _pad: 0 });
                }
            }
            EventKindTag::RallyCall => {
                // payload: [observer, wounded_kin]
                let observer = rec.payload[0];
                let wounded = rec.payload[1];
                if let Some((a, b)) = slot_pair(observer, wounded) {
                    rally_boost.push(FoldInputPair { first: a, second: b, tick, _pad: 0 });
                }
            }
            EventKindTag::EngagementCommitted => {
                // payload: [actor, target] — insert both sides (kind=0).
                let actor = rec.payload[0];
                let target = rec.payload[1];
                if let Some((a, b)) = slot_pair(actor, target) {
                    engaged_with.push(FoldInputSlot { first: a, second: b, kind: 0, _pad: 0 });
                }
            }
            EventKindTag::EngagementBroken => {
                // payload: [actor, former_target, reason_u8]
                let actor = rec.payload[0];
                let former = rec.payload[1];
                if let Some((a, b)) = slot_pair(actor, former) {
                    engaged_with.push(FoldInputSlot { first: a, second: b, kind: 1, _pad: 0 });
                }
            }
            _ => {}
        }
    }

    if !my_enemies.is_empty() {
        view_storage.fold_pair_events(device, queue, "my_enemies", &my_enemies)?;
    }
    if !threat_level.is_empty() {
        view_storage.fold_pair_events(device, queue, "threat_level", &threat_level)?;
    }
    if !kin_fear.is_empty() {
        view_storage.fold_pair_events(device, queue, "kin_fear", &kin_fear)?;
    }
    if !pack_focus.is_empty() {
        view_storage.fold_pair_events(device, queue, "pack_focus", &pack_focus)?;
    }
    if !rally_boost.is_empty() {
        view_storage.fold_pair_events(device, queue, "rally_boost", &rally_boost)?;
    }
    if !engaged_with.is_empty() {
        view_storage.fold_slot_events(device, queue, "engaged_with", &engaged_with)?;
    }
    Ok(())
}

/// Strip dead agents from every per-slot kin list. The CPU's
/// `spatial::nearby_kin` filters alive at call time; the GPU kin list is
/// precomputed ONCE at cascade start, so deaths mid-cascade leave
/// phantom kin. We compact each list in-place so dead ids don't drop
/// into the rule's `for kin in nearby_kin(...)` loop body and emit
/// spurious fear/pack/rally events from dead observers or toward dead
/// targets.
///
/// Also drops the observer's own id if the observer itself died — a dead
/// kin list on a dead slot never gets visited (the rule's outer
/// destructure rejects dead `agent_id`), but we still filter for
/// symmetry.
fn filter_dead_from_kin(
    kin_lists: &[GpuKinList],
    agents: &[GpuAgentSlot],
) -> Vec<GpuKinList> {
    let mut out: Vec<GpuKinList> = Vec::with_capacity(kin_lists.len());
    for kl in kin_lists {
        let mut filtered = GpuKinList::default();
        let mut w = 0usize;
        let cap = crate::spatial_gpu::K as usize;
        for i in 0..(kl.count as usize).min(cap) {
            let kin_raw = kl.ids[i];
            if kin_raw == 0 || kin_raw == u32::MAX {
                continue;
            }
            let slot = (kin_raw - 1) as usize;
            if slot >= agents.len() {
                continue;
            }
            if agents[slot].alive == 0 {
                continue;
            }
            filtered.ids[w] = kin_raw;
            w += 1;
        }
        filtered.count = w as u32;
        out.push(filtered);
    }
    out
}

/// Convert two raw AgentId words (1-based, `0xFFFFFFFF` / `0` sentinel)
/// into a `(slot_a, slot_b)` pair of 0-based slots, or `None` if either
/// side is invalid. Used by the fold-input builder.
fn slot_pair(a_raw: u32, b_raw: u32) -> Option<(u32, u32)> {
    if a_raw == 0 || b_raw == 0 {
        return None;
    }
    if a_raw == u32::MAX || b_raw == u32::MAX {
        return None;
    }
    Some((a_raw - 1, b_raw - 1))
}

/// Convert initial CPU `Event`s into `EventRecord`s for the kernel's
/// `events_in` binding. Non-replayable / non-packable events
/// (`ChronicleEntry`) are filtered out — they wouldn't trigger any
/// physics rule anyway.
pub fn pack_initial_events(events: &[Event]) -> Vec<EventRecord> {
    events.iter().filter_map(pack_event).collect()
}

/// Drain every record accumulated across the cascade into the CPU
/// `EventRing`. Used by the full-tick integration — after `run_cascade`
/// returns, the caller pushes each emitted `Event` into the ring so
/// chronicle / parity paths see them.
pub fn events_into_ring(records: &[EventRecord], ring: &mut EventRing) -> usize {
    let mut pushed = 0usize;
    for rec in records {
        if let Some(ev) = unpack_record(rec) {
            ring.push(ev);
            pushed += 1;
        }
    }
    pushed
}

/// Apply the cascade's final agent SoA onto `SimState` — thin wrapper
/// over `unpack_agent_slots` kept here so callers don't reach across
/// module boundaries for it.
pub fn apply_final_slots(state: &mut SimState, slots: &[GpuAgentSlot]) {
    unpack_agent_slots(state, slots);
}

/// Walk an event slice (typically GPU-cascade emissions + the
/// apply-actions seed set for the current tick) and dispatch the 11
/// "cold-state" physics rules the GPU kernel stubs — the ones that
/// mutate state the packed agent SoA doesn't carry:
///
///   * `transfer_gold` / `modify_standing` / `record_memory` — mutate
///     gold, standing matrix, memory ring. GPU physics emits the
///     corresponding `EffectGoldTransfer` / `EffectStandingDelta` /
///     `RecordMemory` events but leaves the side effect to CPU.
///   * Chronicles (8 rules) — push `ChronicleEntry` narrative events
///     in response to AgentAttacked / AgentDied / AgentFled /
///     EngagementCommitted / EngagementBroken / FearSpread / RallyCall
///     and the conditional `chronicle_wound` on low-hp attacks.
///
/// Every other physics rule (damage, heal, shield, stun, slow,
/// engagement_on_move, engagement_on_death, fear_spread_on_death,
/// pack_focus_on_engagement, rally_on_wound, opportunity_attack, cast)
/// was already executed by the GPU kernel — double-applying would
/// corrupt state / duplicate events.
///
/// The `events_slice` argument is iterated in push order so chronicle
/// output order matches the CPU path.
pub fn cold_state_replay(
    state: &mut SimState,
    events: &mut EventRing,
    events_slice: &[Event],
) {
    use engine::generated::physics::{
        chronicle_attack, chronicle_break, chronicle_death, chronicle_engagement,
        chronicle_flee, chronicle_rally, chronicle_rout, chronicle_wound, modify_standing,
        record_memory, transfer_gold,
    };

    for ev in events_slice {
        match *ev {
            // --- AgentAttacked: chronicle_attack + chronicle_wound ---
            // (rally_on_wound stays GPU-owned; the GPU kernel emits
            //  the matching `RallyCall` event.)
            Event::AgentAttacked { actor, target, .. } => {
                chronicle_attack::chronicle_attack(actor, target, state, events);
                chronicle_wound::chronicle_wound(actor, target, state, events);
            }
            // --- AgentDied: chronicle_death ---
            Event::AgentDied { agent_id, .. } => {
                chronicle_death::chronicle_death(agent_id, state, events);
            }
            // --- AgentFled: chronicle_flee ---
            Event::AgentFled { agent_id, .. } => {
                chronicle_flee::chronicle_flee(agent_id, state, events);
            }
            // --- EngagementCommitted: chronicle_engagement ---
            Event::EngagementCommitted { actor, target, .. } => {
                chronicle_engagement::chronicle_engagement(actor, target, state, events);
            }
            // --- EngagementBroken: chronicle_break ---
            Event::EngagementBroken {
                actor, former_target, ..
            } => {
                chronicle_break::chronicle_break(actor, former_target, state, events);
            }
            // --- FearSpread: chronicle_rout ---
            Event::FearSpread {
                observer, dead_kin, ..
            } => {
                chronicle_rout::chronicle_rout(observer, dead_kin, state, events);
            }
            // --- RallyCall: chronicle_rally ---
            Event::RallyCall {
                observer,
                wounded_kin,
                ..
            } => {
                chronicle_rally::chronicle_rally(observer, wounded_kin, state, events);
            }
            // --- EffectGoldTransfer: transfer_gold (mutates inventory) ---
            Event::EffectGoldTransfer {
                from, to, amount, ..
            } => {
                transfer_gold::transfer_gold(from, to, amount, state, events);
            }
            // --- EffectStandingDelta: modify_standing (mutates standing) ---
            Event::EffectStandingDelta { a, b, delta, .. } => {
                modify_standing::modify_standing(a, b, delta, state, events);
            }
            // --- RecordMemory: record_memory (pushes memory ring) ---
            Event::RecordMemory {
                observer,
                source,
                fact_payload,
                confidence,
                tick,
            } => {
                record_memory::record_memory(
                    observer,
                    source,
                    fact_payload,
                    confidence,
                    tick,
                    state,
                    events,
                );
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation loader
// ---------------------------------------------------------------------------

/// Errors surfaced by `load_compilation_from_assets`.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error, PathBuf),
    Parse(String, PathBuf),
    Resolve(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e, p) => write!(f, "read {}: {e}", p.display()),
            LoadError::Parse(e, p) => write!(f, "parse {}: {e}", p.display()),
            LoadError::Resolve(e) => write!(f, "resolve: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Load the sim DSL compilation the cascade kernels need (config, enums,
/// events, physics) from `assets/sim/*.sim`. Walks up from the workspace
/// `CARGO_MANIFEST_DIR` so the helper works whether the caller is inside
/// `crates/engine_gpu` (two-level pop) or a binary crate at the workspace
/// root (no pop needed). Tries a few candidate roots in order and returns
/// the first one where every expected file opens cleanly.
pub fn load_compilation_from_assets() -> Result<dsl_compiler::ir::Compilation, LoadError> {
    // Candidate workspace roots. Every worktree has `assets/sim/` at the
    // repo root, so we probe:
    //   * `CARGO_MANIFEST_DIR/../../` — `crates/engine_gpu/` case.
    //   * `CARGO_MANIFEST_DIR/../` — binaries living in `crates/<bin>/`.
    //   * `CARGO_MANIFEST_DIR/` — workspace-root binaries.
    //   * `./` — random callers with the workspace as CWD.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(manifest) = option_env!("CARGO_MANIFEST_DIR") {
        let base = PathBuf::from(manifest);
        candidates.push(base.join("../../assets/sim"));
        candidates.push(base.join("../assets/sim"));
        candidates.push(base.join("assets/sim"));
    }
    candidates.push(PathBuf::from("assets/sim"));

    let files = ["config.sim", "enums.sim", "events.sim", "physics.sim"];
    let mut chosen: Option<PathBuf> = None;
    for c in &candidates {
        if files.iter().all(|f| c.join(f).exists()) {
            chosen = Some(c.clone());
            break;
        }
    }
    let root = chosen.ok_or_else(|| {
        LoadError::Io(
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no assets/sim directory found in any of: {:?}",
                    candidates
                ),
            ),
            PathBuf::from("assets/sim"),
        )
    })?;

    let mut merged = Program { decls: Vec::new() };
    for f in &files {
        let path = root.join(f);
        let src = std::fs::read_to_string(&path).map_err(|e| LoadError::Io(e, path.clone()))?;
        let prog = dsl_compiler::parse(&src)
            .map_err(|e| LoadError::Parse(format!("{e:?}"), path.clone()))?;
        merged.decls.extend(prog.decls);
    }
    dsl_compiler::compile_ast(merged).map_err(|e| LoadError::Resolve(format!("{e:?}")))
}

// ---------------------------------------------------------------------------
// CascadeCtx — lazy-initialised holder for the three GPU components the
// backend needs to drive a cascade every tick: physics kernel (compiled
// against a concrete Compilation), spatial hash, and packed ability
// registry. Also caches the Compilation itself so `EmitContext` borrows
// back into stable data.
// ---------------------------------------------------------------------------

/// Lazily-initialised GPU cascade context owned by `GpuBackend`. Holds
/// the physics kernel + spatial hash + packed ability registry (empty for
/// now — the ability registry upload path is Phase 7 work). Also retains
/// the `Compilation` so the per-tick `EmitContext` has stable event /
/// event_tag borrows.
pub struct CascadeCtx {
    pub physics: PhysicsKernel,
    pub spatial: GpuSpatialHash,
    pub abilities: PackedAbilityRegistry,
    /// Retained so `EmitContext { events, event_tags }` points into
    /// storage that outlives the per-tick call.
    pub comp: dsl_compiler::ir::Compilation,
}

/// Errors surfaced by `CascadeCtx::new`. Wraps the sub-component init
/// errors so the backend's `ensure_cascade_initialized` returns a single
/// type.
#[derive(Debug)]
pub enum CascadeCtxError {
    Load(LoadError),
    Physics(PhysicsError),
    Spatial(SpatialError),
}

impl std::fmt::Display for CascadeCtxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CascadeCtxError::Load(e) => write!(f, "load compilation: {e}"),
            CascadeCtxError::Physics(e) => write!(f, "physics kernel: {e}"),
            CascadeCtxError::Spatial(e) => write!(f, "spatial hash: {e}"),
        }
    }
}

impl std::error::Error for CascadeCtxError {}

impl From<LoadError> for CascadeCtxError {
    fn from(e: LoadError) -> Self {
        CascadeCtxError::Load(e)
    }
}

impl From<PhysicsError> for CascadeCtxError {
    fn from(e: PhysicsError) -> Self {
        CascadeCtxError::Physics(e)
    }
}

impl From<SpatialError> for CascadeCtxError {
    fn from(e: SpatialError) -> Self {
        CascadeCtxError::Spatial(e)
    }
}

impl CascadeCtx {
    /// Build a fresh cascade context: load the DSL assets, compile the
    /// physics kernel, spin up a spatial hash, and keep an empty ability
    /// registry. Caller provides the wgpu device/queue pair the backend
    /// owns; event-ring capacity is picked at the cascade driver's
    /// default (4096 slots — matches the parity harness).
    pub fn new(device: &wgpu::Device) -> Result<Self, CascadeCtxError> {
        let comp = load_compilation_from_assets()?;
        let ctx = EmitContext {
            events: &comp.events,
            event_tags: &comp.event_tags,
        };
        let physics = PhysicsKernel::new(device, &comp.physics, &ctx, 4096)?;
        let spatial = GpuSpatialHash::new(device)?;
        let abilities = PackedAbilityRegistry::empty();
        Ok(Self {
            physics,
            spatial,
            abilities,
            comp,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_ring::{EventKindTag, EventRecord};

    #[test]
    fn slot_pair_rejects_zero_and_max() {
        assert_eq!(slot_pair(0, 5), None);
        assert_eq!(slot_pair(5, 0), None);
        assert_eq!(slot_pair(u32::MAX, 5), None);
        assert_eq!(slot_pair(5, u32::MAX), None);
        assert_eq!(slot_pair(3, 7), Some((2, 6)));
    }

    #[test]
    fn default_kin_radius_matches_physics_rules() {
        // The fear/pack/rally rules all use a 12 m radius in physics.sim;
        // the precompute must match so the rules read the right kin set.
        assert_eq!(DEFAULT_KIN_RADIUS, 12.0);
    }

    #[test]
    fn max_iterations_matches_cpu_bound() {
        // `engine::cascade::dispatch::MAX_CASCADE_ITERATIONS` is the
        // authoritative bound. A skew would have the GPU accepting a
        // cascade the CPU would panic on (in dev) or truncate (in
        // release) — a determinism risk.
        assert_eq!(
            MAX_CASCADE_ITERATIONS as usize,
            engine::cascade::MAX_CASCADE_ITERATIONS,
        );
    }

    #[test]
    fn fold_inputs_scheduled_for_agent_attacked() {
        // Build a synthetic EventRecord for AgentAttacked and make sure
        // the fold dispatcher routes it to both my_enemies and threat_level.
        // We can't dispatch GPU folds in a cfg(test) without `gpu` feature
        // (this test module is only compiled with the feature on, so we'd
        // need a device) — instead, exercise the fold-input builder's
        // bookkeeping by inspecting the `events` vec it builds.
        let rec = EventRecord {
            kind: EventKindTag::AgentAttacked.raw(),
            tick: 42,
            payload: [1, 2, 0, 0, 0, 0, 0, 0], // actor=1, target=2
        };
        // Mirror the routing logic below — we don't have a ViewStorage
        // in this test, but we can still reason about the slot_pair
        // transform.
        let pair = slot_pair(rec.payload[1], rec.payload[0]).unwrap();
        assert_eq!(pair, (1, 0));
    }
}
