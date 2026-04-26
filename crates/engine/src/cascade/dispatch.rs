use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{EventLike, EventRing};
use crate::state::SimState;
use crate::telemetry::{metrics, TelemetrySink};

// ---------------------------------------------------------------------------
// Interpreted-rules cascade dispatch helpers (feature = "interpreted-rules")
// ---------------------------------------------------------------------------
//
// When the `interpreted-rules` feature is active, physics/cascade handlers
// are driven by `dsl_ast::eval::physics::PhysicsIR::apply` instead of the
// compiler-emitted `kind_dispatchers`. Legacy trait-object handlers (table)
// still fire after the interpreter, matching the compiled-path semantics.
//
// The interpreter dispatch is generic over `E: EventLike` via an optional
// function pointer stored in `CascadeRegistry`. For the concrete
// `engine_data::events::Event` instantiation, the function pointer is
// installed by `engine_rules::lib::with_engine_builtins` via
// `CascadeRegistry::set_interp_dispatch`. For other event types the field
// stays `None` (zero cost, no-op in the dispatch hot path).

#[cfg(feature = "interpreted-rules")]
mod interp {
    use engine_data::events::Event;
    use crate::evaluator::context::EngineCascadeCtx;
    use crate::event::EventRing;
    use crate::state::SimState;
    use dsl_ast::eval::{AbilityId as DslAbilityId, AgentId as DslAgentId, EvalValue};

    // ---- ID bridging helpers -----------------------------------------------

    #[inline]
    fn dsl_agent(id: crate::ids::AgentId) -> DslAgentId {
        DslAgentId::new(id.raw()).expect("engine AgentId is always non-zero")
    }

    #[inline]
    fn dsl_ability(id: crate::ids::AbilityId) -> DslAbilityId {
        DslAbilityId::new(id.raw()).expect("engine AbilityId is always non-zero")
    }

    // ---- Event → kind-name string -----------------------------------------

    /// Return the canonical DSL event-name string for this event, used to
    /// match against `PhysicsHandlerIR::pattern.display_name()`.
    pub fn event_kind_name(event: &Event) -> &'static str {
        match event {
            Event::AgentMoved            { .. } => "AgentMoved",
            Event::AgentAttacked         { .. } => "AgentAttacked",
            Event::AgentDied             { .. } => "AgentDied",
            Event::AgentFled             { .. } => "AgentFled",
            Event::AgentAte              { .. } => "AgentAte",
            Event::AgentDrank            { .. } => "AgentDrank",
            Event::AgentRested           { .. } => "AgentRested",
            Event::AgentCast             { .. } => "AgentCast",
            Event::AgentUsedItem         { .. } => "AgentUsedItem",
            Event::AgentHarvested        { .. } => "AgentHarvested",
            Event::AgentPlacedTile       { .. } => "AgentPlacedTile",
            Event::AgentPlacedVoxel      { .. } => "AgentPlacedVoxel",
            Event::AgentHarvestedVoxel   { .. } => "AgentHarvestedVoxel",
            Event::AgentConversed        { .. } => "AgentConversed",
            Event::AgentSharedStory      { .. } => "AgentSharedStory",
            Event::AgentCommunicated     { .. } => "AgentCommunicated",
            Event::InformationRequested  { .. } => "InformationRequested",
            Event::AgentRemembered       { .. } => "AgentRemembered",
            Event::QuestPosted           { .. } => "QuestPosted",
            Event::QuestAccepted         { .. } => "QuestAccepted",
            Event::BidPlaced             { .. } => "BidPlaced",
            Event::AnnounceEmitted       { .. } => "AnnounceEmitted",
            Event::RecordMemory          { .. } => "RecordMemory",
            Event::OpportunityAttackTriggered { .. } => "OpportunityAttackTriggered",
            Event::EffectDamageApplied   { .. } => "EffectDamageApplied",
            Event::EffectHealApplied     { .. } => "EffectHealApplied",
            Event::EffectShieldApplied   { .. } => "EffectShieldApplied",
            Event::EffectStunApplied     { .. } => "EffectStunApplied",
            Event::EffectSlowApplied     { .. } => "EffectSlowApplied",
            Event::EffectGoldTransfer    { .. } => "EffectGoldTransfer",
            Event::EffectStandingDelta   { .. } => "EffectStandingDelta",
            Event::CastDepthExceeded     { .. } => "CastDepthExceeded",
            Event::EngagementCommitted   { .. } => "EngagementCommitted",
            Event::EngagementBroken      { .. } => "EngagementBroken",
            Event::FearSpread            { .. } => "FearSpread",
            Event::PackAssist            { .. } => "PackAssist",
            Event::RallyCall             { .. } => "RallyCall",
            Event::ChronicleEntry        { .. } => "ChronicleEntry",
        }
    }

    // ---- Event → (name, EvalValue) field slice ----------------------------

    /// Unpack every named field of an `Event` variant into a
    /// `Vec<(&'static str, EvalValue)>` suitable for passing to
    /// `PhysicsIR::apply`. One match arm per variant; all other variants
    /// produce an empty vec (no physics rule matches them).
    pub fn event_to_fields(event: &Event) -> Vec<(&'static str, EvalValue)> {
        match event {
            // ---- AgentMoved -----------------------------------------------
            Event::AgentMoved { actor, from, location, tick } => vec![
                ("actor",    EvalValue::Agent(dsl_agent(*actor))),
                ("from",     EvalValue::F32(from.x)),
                ("location", EvalValue::F32(location.x)),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- AgentAttacked --------------------------------------------
            Event::AgentAttacked { actor, target, damage, tick } => vec![
                ("actor",  EvalValue::Agent(dsl_agent(*actor))),
                ("target", EvalValue::Agent(dsl_agent(*target))),
                ("damage", EvalValue::F32(*damage)),
                ("a",      EvalValue::Agent(dsl_agent(*actor))),
                ("t",      EvalValue::Agent(dsl_agent(*target))),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- AgentDied ------------------------------------------------
            Event::AgentDied { agent_id, tick } => vec![
                ("agent_id", EvalValue::Agent(dsl_agent(*agent_id))),
                ("agent",    EvalValue::Agent(dsl_agent(*agent_id))),
                ("a",        EvalValue::Agent(dsl_agent(*agent_id))),
                ("dead",     EvalValue::Agent(dsl_agent(*agent_id))),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- AgentFled ------------------------------------------------
            Event::AgentFled { agent_id, tick, .. } => vec![
                ("agent_id", EvalValue::Agent(dsl_agent(*agent_id))),
                ("a",        EvalValue::Agent(dsl_agent(*agent_id))),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- AgentCast ------------------------------------------------
            Event::AgentCast { actor, ability, target, depth, tick } => vec![
                ("actor",   EvalValue::Agent(dsl_agent(*actor))),
                ("ability", EvalValue::Ability(dsl_ability(*ability))),
                ("target",  EvalValue::Agent(dsl_agent(*target))),
                ("depth",   EvalValue::U32(*depth as u32)),
                ("caster",  EvalValue::Agent(dsl_agent(*actor))),
                ("ab",      EvalValue::Ability(dsl_ability(*ability))),
                ("t",       EvalValue::U32(*tick)),
                ("tick",    EvalValue::U32(*tick)),
            ],
            // ---- EffectDamageApplied --------------------------------------
            Event::EffectDamageApplied { actor, target, amount, tick } => vec![
                ("actor",  EvalValue::Agent(dsl_agent(*actor))),
                ("target", EvalValue::Agent(dsl_agent(*target))),
                ("amount", EvalValue::F32(*amount)),
                ("c",      EvalValue::Agent(dsl_agent(*actor))),
                ("t",      EvalValue::Agent(dsl_agent(*target))),
                ("a",      EvalValue::F32(*amount)),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- EffectHealApplied ----------------------------------------
            Event::EffectHealApplied { actor, target, amount, tick } => vec![
                ("actor",  EvalValue::Agent(dsl_agent(*actor))),
                ("target", EvalValue::Agent(dsl_agent(*target))),
                ("amount", EvalValue::F32(*amount)),
                ("c",      EvalValue::Agent(dsl_agent(*actor))),
                ("t",      EvalValue::Agent(dsl_agent(*target))),
                ("a",      EvalValue::F32(*amount)),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- EffectShieldApplied --------------------------------------
            Event::EffectShieldApplied { actor, target, amount, tick } => vec![
                ("actor",  EvalValue::Agent(dsl_agent(*actor))),
                ("target", EvalValue::Agent(dsl_agent(*target))),
                ("amount", EvalValue::F32(*amount)),
                ("c",      EvalValue::Agent(dsl_agent(*actor))),
                ("t",      EvalValue::Agent(dsl_agent(*target))),
                ("a",      EvalValue::F32(*amount)),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- EffectStunApplied ----------------------------------------
            Event::EffectStunApplied { actor, target, expires_at_tick, tick } => vec![
                ("actor",           EvalValue::Agent(dsl_agent(*actor))),
                ("target",          EvalValue::Agent(dsl_agent(*target))),
                ("expires_at_tick", EvalValue::U32(*expires_at_tick)),
                ("c",               EvalValue::Agent(dsl_agent(*actor))),
                ("t",               EvalValue::Agent(dsl_agent(*target))),
                ("e",               EvalValue::U32(*expires_at_tick)),
                ("tick",            EvalValue::U32(*tick)),
            ],
            // ---- EffectSlowApplied ----------------------------------------
            Event::EffectSlowApplied { actor, target, expires_at_tick, factor_q8, tick } => vec![
                ("actor",           EvalValue::Agent(dsl_agent(*actor))),
                ("target",          EvalValue::Agent(dsl_agent(*target))),
                ("expires_at_tick", EvalValue::U32(*expires_at_tick)),
                ("factor_q8",       EvalValue::I32(*factor_q8 as i32)),
                ("c",               EvalValue::Agent(dsl_agent(*actor))),
                ("t",               EvalValue::Agent(dsl_agent(*target))),
                ("e",               EvalValue::U32(*expires_at_tick)),
                ("f",               EvalValue::I32(*factor_q8 as i32)),
                ("tick",            EvalValue::U32(*tick)),
            ],
            // ---- EffectGoldTransfer ---------------------------------------
            Event::EffectGoldTransfer { from, to, amount, tick } => vec![
                ("from",   EvalValue::Agent(dsl_agent(*from))),
                ("to",     EvalValue::Agent(dsl_agent(*to))),
                ("amount", EvalValue::I64(*amount as i64)),
                ("a",      EvalValue::I64(*amount as i64)),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- EffectStandingDelta --------------------------------------
            Event::EffectStandingDelta { a, b, delta, tick } => vec![
                ("a",     EvalValue::Agent(dsl_agent(*a))),
                ("b",     EvalValue::Agent(dsl_agent(*b))),
                ("delta", EvalValue::I32(*delta as i32)),
                ("tick",  EvalValue::U32(*tick)),
            ],
            // ---- OpportunityAttackTriggered -------------------------------
            Event::OpportunityAttackTriggered { actor, target, tick } => vec![
                ("actor",    EvalValue::Agent(dsl_agent(*actor))),
                ("target",   EvalValue::Agent(dsl_agent(*target))),
                ("attacker", EvalValue::Agent(dsl_agent(*actor))),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- RecordMemory ---------------------------------------------
            Event::RecordMemory { observer, source, fact_payload, confidence, tick } => vec![
                ("observer",     EvalValue::Agent(dsl_agent(*observer))),
                ("source",       EvalValue::Agent(dsl_agent(*source))),
                ("fact_payload", EvalValue::I64(*fact_payload as i64)),
                ("confidence",   EvalValue::F32(*confidence)),
                ("o",            EvalValue::Agent(dsl_agent(*observer))),
                ("s",            EvalValue::Agent(dsl_agent(*source))),
                ("f",            EvalValue::I64(*fact_payload as i64)),
                ("c",            EvalValue::F32(*confidence)),
                ("t",            EvalValue::U32(*tick)),
                ("tick",         EvalValue::U32(*tick)),
            ],
            // ---- EngagementCommitted --------------------------------------
            Event::EngagementCommitted { actor, target, tick } => vec![
                ("actor",  EvalValue::Agent(dsl_agent(*actor))),
                ("target", EvalValue::Agent(dsl_agent(*target))),
                ("a",      EvalValue::Agent(dsl_agent(*actor))),
                ("t",      EvalValue::Agent(dsl_agent(*target))),
                ("tick",   EvalValue::U32(*tick)),
            ],
            // ---- EngagementBroken -----------------------------------------
            Event::EngagementBroken { actor, former_target, reason, tick } => vec![
                ("actor",         EvalValue::Agent(dsl_agent(*actor))),
                ("former_target", EvalValue::Agent(dsl_agent(*former_target))),
                ("reason",        EvalValue::U32(*reason as u32)),
                ("a",             EvalValue::Agent(dsl_agent(*actor))),
                ("t",             EvalValue::Agent(dsl_agent(*former_target))),
                ("tick",          EvalValue::U32(*tick)),
            ],
            // ---- FearSpread -----------------------------------------------
            Event::FearSpread { observer, dead_kin, tick } => vec![
                ("observer", EvalValue::Agent(dsl_agent(*observer))),
                ("dead_kin", EvalValue::Agent(dsl_agent(*dead_kin))),
                ("o",        EvalValue::Agent(dsl_agent(*observer))),
                ("d",        EvalValue::Agent(dsl_agent(*dead_kin))),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- PackAssist -----------------------------------------------
            Event::PackAssist { observer, target, tick } => vec![
                ("observer", EvalValue::Agent(dsl_agent(*observer))),
                ("target",   EvalValue::Agent(dsl_agent(*target))),
                ("tick",     EvalValue::U32(*tick)),
            ],
            // ---- RallyCall ------------------------------------------------
            Event::RallyCall { observer, wounded_kin, tick } => vec![
                ("observer",    EvalValue::Agent(dsl_agent(*observer))),
                ("wounded_kin", EvalValue::Agent(dsl_agent(*wounded_kin))),
                ("o",           EvalValue::Agent(dsl_agent(*observer))),
                ("w",           EvalValue::Agent(dsl_agent(*wounded_kin))),
                ("tick",        EvalValue::U32(*tick)),
            ],
            // ---- ChronicleEntry -------------------------------------------
            Event::ChronicleEntry { template_id, agent, target, tick } => vec![
                ("template_id", EvalValue::U32(*template_id)),
                ("agent",       EvalValue::Agent(dsl_agent(*agent))),
                ("target",      EvalValue::Agent(dsl_agent(*target))),
                ("tick",        EvalValue::U32(*tick)),
            ],
            // All other variants produce no fields.
            _ => vec![],
        }
    }

    // ---- Interpreted physics dispatch -------------------------------------

    /// Dispatch `event` through all `PhysicsIR` handlers in the compiled DSL
    /// whose `on` pattern matches the event's kind name.
    ///
    /// Ordering: iterates `comp.physics` in DSL source order, which matches
    /// the compiled `register()` call order in `engine_rules/src/physics/`.
    /// Handlers for the same event kind fire in the same sequence as the
    /// compiled path, preserving cascade semantics.
    pub fn dispatch_interpreted(event: &Event, state: &mut SimState, events: &mut EventRing<Event>) {
        let kind_name = event_kind_name(event);
        let event_fields = event_to_fields(event);
        let comp = crate::mask::interp::compilation();

        for physics in &comp.physics {
            for (idx, handler) in physics.handlers.iter().enumerate() {
                if handler.pattern.display_name() == kind_name {
                    let mut ctx = EngineCascadeCtx::new(state, events);
                    physics.apply(idx, &event_fields, &mut ctx);
                }
            }
        }
    }
}

/// Public free-function entry point for the interpreter cascade dispatch hook.
///
/// This is the function pointer installed via `CascadeRegistry::set_interp_dispatch`
/// in `engine_rules::with_engine_builtins` when the `interpreted-rules` feature
/// is active. It has the exact signature required by `InterpDispatchFn<Event>`.
#[cfg(feature = "interpreted-rules")]
pub fn interp_dispatch_hook(
    event: &engine_data::events::Event,
    state: &mut crate::state::SimState,
    events: &mut crate::event::EventRing<engine_data::events::Event>,
) {
    interp::dispatch_interpreted(event, state, events);
}

/// Dense slot count covering all `EventKindId` ordinals — includes the 128+
/// chronicle reservation. `Vec<Vec<Box<dyn CascadeHandler>>>` indexed by
/// `[lane as usize][kind as u8 as usize]`.
const KIND_SLOTS: usize = 256;

/// Maximum number of cascade dispatch passes per `run_fixed_point` call.
/// If handlers keep pushing new events beyond this bound, the cascade is
/// considered non-converging: dev builds panic, release builds log and
/// truncate.
pub const MAX_CASCADE_ITERATIONS: usize = 8;

/// Type signature of a compiler-emitted per-event-kind dispatcher. The
/// dispatcher destructures the triggering event once and fans the call
/// out to every applicable handler (kind-specific + tag-matched).
/// The `V` parameter is the views type threaded through all handlers.
pub type KindDispatcher<E, V> = fn(&E, &mut SimState, &mut V, &mut EventRing<E>);

pub struct CascadeRegistry<E: EventLike, V = ()> {
    table: Vec<Vec<Vec<Box<dyn __object_safe::DynHandler<E, V>>>>>,
    /// Compiler-emitted per-event-kind dispatcher fns. Indexed by
    /// `EventKindId as u8 as usize`; `None` means no dispatcher is
    /// installed for that kind (falls back to per-handler trait-object
    /// dispatch via `table`).
    kind_dispatchers: Vec<Option<KindDispatcher<E, V>>>,
    /// Optional interpreter dispatch hook (feature = "interpreted-rules").
    ///
    /// When installed (non-None), this function is called instead of the
    /// `kind_dispatchers` fast path. The legacy trait-object `table` still
    /// runs after it. Installed by `engine_rules::with_engine_builtins` for
    /// the concrete `Event` instantiation; remains `None` for other `E` types
    /// and when the feature is off (zero cost).
    #[cfg(feature = "interpreted-rules")]
    interp_dispatch: Option<fn(&E, &mut SimState, &mut EventRing<E>)>,
}

/// Object-safe wrapper trait used for the boxed handler table.
/// Not public — internal to the dispatch machinery.
mod __object_safe {
    use crate::event::{EventLike, EventRing};
    use crate::state::SimState;

    pub trait DynHandler<E: EventLike, V>: Send + Sync {
        #[allow(dead_code)]
        fn trigger_kind(&self) -> u8;
        #[allow(dead_code)]
        fn lane_ord(&self) -> u8;
        fn handle_dyn(&self, event: &E, state: &mut SimState, views: &mut V, events: &mut EventRing<E>);
    }

    impl<E: EventLike, H: super::super::handler::CascadeHandler<E>> DynHandler<E, H::Views>
        for H
    where
        H: Send + Sync,
    {
        fn trigger_kind(&self) -> u8 { self.trigger() as u8 }
        fn lane_ord(&self) -> u8 { self.lane() as u8 }
        fn handle_dyn(&self, event: &E, state: &mut SimState, views: &mut H::Views, events: &mut EventRing<E>) {
            self.handle(event, state, views, events);
        }
    }
}

impl<E: EventLike, V> CascadeRegistry<E, V> {
    pub fn new() -> Self {
        Self {
            table: (0..Lane::ALL.len()).map(|_| (0..KIND_SLOTS).map(|_| Vec::new()).collect()).collect(),
            kind_dispatchers: (0..KIND_SLOTS).map(|_| None).collect(),
            #[cfg(feature = "interpreted-rules")]
            interp_dispatch: None,
        }
    }

    /// Install an interpreter dispatch hook for the `interpreted-rules` path.
    ///
    /// When installed, the hook is called in place of the compiler-emitted
    /// `kind_dispatchers` during `dispatch`. Legacy trait-object handlers
    /// in `table` still fire after it. Only has effect when the
    /// `interpreted-rules` feature is enabled; the parameter is ignored
    /// (and the method becomes a no-op) otherwise.
    #[cfg(feature = "interpreted-rules")]
    pub fn set_interp_dispatch(&mut self, f: fn(&E, &mut SimState, &mut EventRing<E>)) {
        self.interp_dispatch = Some(f);
    }
    #[cfg(not(feature = "interpreted-rules"))]
    #[allow(unused_variables)]
    pub fn set_interp_dispatch(&mut self, _f: fn(&E, &mut SimState, &mut EventRing<E>)) {}


    /// Install a compiler-emitted per-event-kind dispatcher. Overwrites
    /// any previously installed dispatcher for the same kind — the DSL
    /// emitter produces one dispatcher per event kind, so reinstallation
    /// is idempotent within a single registration call.
    pub fn install_kind(&mut self, kind: EventKindId, dispatcher: KindDispatcher<E, V>) {
        let idx = kind as u8 as usize;
        self.kind_dispatchers[idx] = Some(dispatcher);
    }

    pub fn register<H: CascadeHandler<E, Views = V> + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &E, state: &mut SimState, views: &mut V, events: &mut EventRing<E>) {
        let kind = event.kind() as u8 as usize;

        // ---------------------------------------------------------------------------
        // Interpreted-rules dispatch path (feature = "interpreted-rules")
        //
        // When the feature is active and an interp hook is installed, the
        // compiler-emitted `kind_dispatchers` are skipped. The interpreter hook
        // calls `PhysicsIR::apply` for every matching DSL physics handler in
        // source order (same as `register()` call order). Legacy trait-object
        // handlers in `table` still run after, matching the compiled-path semantics
        // where `kind_dispatchers` is followed by the `table` walk.
        // ---------------------------------------------------------------------------
        #[cfg(feature = "interpreted-rules")]
        if let Some(hook) = self.interp_dispatch {
            hook(event, state, events);
        }

        #[cfg(not(feature = "interpreted-rules"))]
        // Prefer the compiler-emitted per-kind dispatcher when installed.
        // It fans out to every applicable handler (kind-specific +
        // tag-matched) inline — no runtime handler-list walk.
        if let Some(dispatcher) = self.kind_dispatchers[kind] {
            dispatcher(event, state, views, events);
        }

        // Legacy trait-object handlers still register via `register`;
        // walk them in lane order after the flat dispatcher.
        for lane in Lane::ALL {
            for handler in &self.table[*lane as usize][kind] {
                handler.handle_dyn(event, state, views, events);
            }
        }
    }

    /// Dispatch any events pushed to `events` that haven't been dispatched yet,
    /// iterating until no new events are emitted, bounded by
    /// `MAX_CASCADE_ITERATIONS`. In dev builds non-convergence panics; in
    /// release it logs and truncates.
    ///
    /// Uses the ring's persistent `dispatched` cursor so multiple calls (e.g.
    /// one per tick) don't re-dispatch past events. Within a single call,
    /// iteration continues as long as handlers push new events, up to the
    /// iteration bound.
    ///
    /// Back-compat wrapper over [`run_fixed_point_tel`] for call sites that
    /// don't have a telemetry sink (typically tests).
    pub fn run_fixed_point(&self, state: &mut SimState, views: &mut V, events: &mut EventRing<E>) {
        self.run_fixed_point_tel(state, views, events, &crate::telemetry::NullSink);
    }

    /// Like [`run_fixed_point`] but also emits the
    /// `metrics::CASCADE_ITERATIONS` histogram metric once per call, counting
    /// the number of dispatch passes taken (0 when the initial ring cursor is
    /// already current — no events to drain). Audit fix HIGH #5.
    pub fn run_fixed_point_tel(
        &self,
        state:     &mut SimState,
        views:     &mut V,
        events:    &mut EventRing<E>,
        telemetry: &dyn TelemetrySink,
    ) {
        let mut processed = events.dispatched();
        let mut iterations: usize = 0;
        for iter in 0..MAX_CASCADE_ITERATIONS {
            let snapshot = events.total_pushed();
            if snapshot == processed {
                events.set_dispatched(processed);
                telemetry.emit_histogram(metrics::CASCADE_ITERATIONS, iterations as f64);
                return;
            }
            iterations = iter + 1;
            for idx in processed..snapshot {
                if let Some(e) = events.get_pushed(idx) {
                    self.dispatch(&e, state, views, events);
                }
            }
            processed = snapshot;
            if iter == MAX_CASCADE_ITERATIONS - 1 {
                // Check again — if handlers emitted MORE events in the last pass,
                // we're about to truncate.
                if events.total_pushed() > processed {
                    #[cfg(debug_assertions)]
                    panic!(
                        "cascade did not converge within {} iterations (tick pushes: {} → {})",
                        MAX_CASCADE_ITERATIONS, processed, events.total_pushed()
                    );
                    #[cfg(not(debug_assertions))]
                    eprintln!(
                        "cascade truncated at {} iterations",
                        MAX_CASCADE_ITERATIONS,
                    );
                }
            }
        }
        events.set_dispatched(events.total_pushed());
        telemetry.emit_histogram(metrics::CASCADE_ITERATIONS, iterations as f64);
    }
}

impl<E: EventLike, V> Default for CascadeRegistry<E, V> {
    fn default() -> Self { Self::new() }
}

// `with_engine_builtins` was deleted along with engine/src/generated/. The
// replacement is compiler-emitted into engine_rules/src/cascade.rs (Task 11
// of Plan B1').
