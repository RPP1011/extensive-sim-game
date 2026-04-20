// crates/engine/src/mask.rs
use crate::ability::evaluate_cast_gate;
use crate::generated::mask::{
    mask_attack_candidates, mask_drink, mask_eat, mask_flee, mask_hold, mask_move_toward_candidates,
    mask_rest,
};
use crate::ids::AgentId;
use crate::state::SimState;
use smallvec::SmallVec;

pub const TARGET_SLOTS: usize = 12;  // matches nearby_actors K=12 per spec §9 D5

// `AGGRO_RANGE` (flee-threat radius) and the attack spatial iterator floor
// now live in `assets/sim/config.sim` as `config.combat.aggro_range` and
// `config.combat.attack_range` respectively. Callers below read them off
// `state.config.combat.*`.

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MicroKind {
    // Movement (3)
    Hold        = 0,
    MoveToward  = 1,
    Flee        = 2,
    // Combat (3)
    Attack      = 3,
    Cast        = 4,
    UseItem     = 5,
    // Resource (4)
    Harvest     = 6,
    Eat         = 7,
    Drink       = 8,
    Rest        = 9,
    // Construction (3)
    PlaceTile    = 10,
    PlaceVoxel   = 11,
    HarvestVoxel = 12,
    // Social (2)
    Converse     = 13,
    ShareStory   = 14,
    // Info push + pull (2)
    Communicate  = 15,
    Ask          = 16,
    // Memory (1)
    Remember     = 17,
}

impl MicroKind {
    pub const ALL: &'static [MicroKind] = &[
        MicroKind::Hold,         MicroKind::MoveToward,   MicroKind::Flee,
        MicroKind::Attack,       MicroKind::Cast,         MicroKind::UseItem,
        MicroKind::Harvest,      MicroKind::Eat,          MicroKind::Drink,
        MicroKind::Rest,         MicroKind::PlaceTile,    MicroKind::PlaceVoxel,
        MicroKind::HarvestVoxel, MicroKind::Converse,     MicroKind::ShareStory,
        MicroKind::Communicate,  MicroKind::Ask,          MicroKind::Remember,
    ];

    /// Target-bound kinds that consume a `TargetMask` candidate list (task
    /// 138). Self-only kinds (Hold / Flee / Eat / Drink / Rest / … the
    /// zero-arg domain hooks) read no candidates and are scored once per
    /// agent. Attack and MoveToward are the only target-bound kinds at
    /// v1 — `Cast`'s target is still inferred via `inferred_cast_target`
    /// in mask-build, and the other targeted heads (Converse /
    /// Communicate / Ask / …) stay permissive until their DSL mask
    /// declarations land.
    pub const TARGET_BOUND: &'static [MicroKind] = &[
        MicroKind::Attack,
        MicroKind::MoveToward,
    ];

    /// Dense 0-based index into `TARGET_BOUND` for this kind, or `None` for
    /// self-only kinds. Used by `TargetMask` to address per-kind candidate
    /// lists.
    pub fn target_slot(self) -> Option<usize> {
        match self {
            MicroKind::Attack => Some(0),
            MicroKind::MoveToward => Some(1),
            _ => None,
        }
    }
}

/// Per-agent, per-target-bound-kind list of valid targets. Task 138 — the
/// scorer argmaxes over these candidate lists for every targeted
/// `MicroKind`, replacing the old `nearest_other` heuristic. Self-only
/// kinds (Hold / Flee / Eat / …) read no candidates and are scored once
/// per agent.
///
/// Storage shape: flat `Vec<SmallVec<...>>` indexed by
/// `agent_slot * TARGET_BOUND.len() + kind.target_slot()`. Inline
/// capacity is 8 per (agent, kind) pair — enough for a small melee mob
/// without spilling to the heap in the common case.
pub struct TargetMask {
    pub candidates: Vec<SmallVec<[AgentId; 8]>>,
}

impl TargetMask {
    pub fn new(n_agents: usize) -> Self {
        let slots = n_agents * MicroKind::TARGET_BOUND.len();
        let mut candidates = Vec::with_capacity(slots);
        for _ in 0..slots {
            candidates.push(SmallVec::new());
        }
        Self { candidates }
    }

    /// Clear every per-agent candidate list; called once per tick before
    /// the target-mask-build phase re-populates them.
    pub fn reset(&mut self) {
        for c in self.candidates.iter_mut() {
            c.clear();
        }
    }

    fn slot(&self, agent: AgentId, kind: MicroKind) -> Option<usize> {
        let agent_slot = (agent.raw() - 1) as usize;
        let kind_slot = kind.target_slot()?;
        Some(agent_slot * MicroKind::TARGET_BOUND.len() + kind_slot)
    }

    /// Push a candidate target into the list for `(agent, kind)`. Called
    /// by the compiler-emitted `mask_<name>_candidates` fns.
    pub fn push(&mut self, agent: AgentId, kind: MicroKind, target: AgentId) {
        if let Some(i) = self.slot(agent, kind) {
            if let Some(list) = self.candidates.get_mut(i) {
                list.push(target);
            }
        }
    }

    /// Read the candidate list for `(agent, kind)`. Empty for self-only
    /// kinds and for (agent, targeted-kind) pairs with no hits.
    pub fn candidates_for(&self, agent: AgentId, kind: MicroKind) -> &[AgentId] {
        match self.slot(agent, kind) {
            Some(i) => self.candidates.get(i).map_or(&[], |v| v.as_slice()),
            None => &[],
        }
    }
}

pub struct MaskBuffer {
    pub micro_kind: Vec<bool>,     // [N_agents × NUM_MICRO]
    pub target:     Vec<bool>,     // [N_agents × TARGET_SLOTS]
}

impl MaskBuffer {
    pub fn new(n_agents: usize) -> Self {
        Self {
            micro_kind: vec![false; n_agents * MicroKind::ALL.len()],
            target:     vec![false; n_agents * TARGET_SLOTS],
        }
    }
    pub fn reset(&mut self) {
        self.micro_kind.iter_mut().for_each(|b| *b = false);
        self.target.iter_mut().for_each(|b| *b = false);
    }
    /// Walk every alive agent and set the bit for `kind` to the value
    /// returned by the DSL-emitted self-predicate `pred(state, self_id)`.
    ///
    /// Task 141 retired the bespoke per-kind `mark_*` bodies (Hold /
    /// MoveToward / Flee / Eat / Drink / Rest) in favour of compiler-
    /// emitted predicates in `crates/engine/src/generated/mask/*.rs`.
    /// Centralising the loop keeps `step::step_full` readable — one
    /// `mark_self_predicate` call per kind — and makes adding a new
    /// self-only mask a one-line engine change once the DSL row lands.
    pub fn mark_self_predicate(
        &mut self,
        state: &SimState,
        kind: MicroKind,
        pred: fn(&SimState, AgentId) -> bool,
    ) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let offset = slot * n_kinds + kind as usize;
            self.micro_kind[offset] = pred(state, id);
        }
    }

    /// Mark `Hold` via the DSL-emitted `mask_hold` predicate.
    pub fn mark_hold_allowed(&mut self, state: &SimState) {
        self.mark_self_predicate(state, MicroKind::Hold, mask_hold);
    }

    /// Mark `MoveToward` as allowed for every agent whose
    /// `mask_move_toward_candidates` enumerator produces at least one
    /// target. Task 138 — MoveToward is now a target-bound kind with a
    /// `from` clause in DSL; this routine populates both the categorical
    /// bit (at least one candidate exists) and the per-agent target
    /// candidate list in `target_mask`.
    pub fn mark_move_allowed_from_candidates(
        &mut self,
        state: &SimState,
        target_mask: &mut TargetMask,
    ) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            mask_move_toward_candidates(state, id, target_mask);
            let has_target = !target_mask.candidates_for(id, MicroKind::MoveToward).is_empty();
            if has_target {
                let slot = (id.raw() - 1) as usize;
                let offset = slot * n_kinds + MicroKind::MoveToward as usize;
                self.micro_kind[offset] = true;
            }
        }
    }

    /// Mark `Flee` via the DSL-emitted `mask_flee` predicate. The DSL
    /// predicate is permissive (allowed for any alive agent) — the real
    /// gate (absolute-hp thresholds `self.hp < 30/50`) lives in the
    /// `Flee` scoring row. Task 138 retired the engine-side "threat
    /// within aggro range" quantifier; Flee stays self-only on the
    /// mask/scoring side and `UtilityBackend::build_action` resolves
    /// the threat (nearest hostile within `config.combat.aggro_range`)
    /// when assembling the `Micro { Flee, Agent(threat) }` action, which
    /// `step_full`'s Flee arm uses to move the agent AWAY from. Task 148.
    pub fn mark_flee_allowed_if_threat_exists(&mut self, state: &SimState) {
        self.mark_self_predicate(state, MicroKind::Flee, mask_flee);
    }

    /// Mark `Eat`, `Drink`, and `Rest` via their DSL-emitted predicates.
    /// Each is self-only and currently unconditional on alive agents,
    /// matching the legacy `mark_needs_allowed` permissiveness.
    pub fn mark_needs_allowed(&mut self, state: &SimState) {
        self.mark_self_predicate(state, MicroKind::Eat, mask_eat);
        self.mark_self_predicate(state, MicroKind::Drink, mask_drink);
        self.mark_self_predicate(state, MicroKind::Rest, mask_rest);
    }

    /// Mark `Attack` via the compiler-emitted candidate enumerator
    /// `mask_attack_candidates`. Task 138 — both the categorical bit
    /// AND the per-agent target candidate list are populated from the
    /// DSL-declared `from query.nearby_agents(...)` source + `when`
    /// predicate. The scorer then argmaxes over the candidate list
    /// rather than resolving a single target via `nearest_other`.
    pub fn mark_attack_allowed_from_candidates(
        &mut self,
        state: &SimState,
        target_mask: &mut TargetMask,
    ) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            mask_attack_candidates(state, id, target_mask);
            let has_target = !target_mask.candidates_for(id, MicroKind::Attack).is_empty();
            if has_target {
                let slot = (id.raw() - 1) as usize;
                let offset = slot * n_kinds + MicroKind::Attack as usize;
                self.micro_kind[offset] = true;
            }
        }
    }

    /// MVP permissiveness: unconditionally allow all 11 domain-hook event-only
    /// micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse,
    /// ShareStory, Communicate, Ask, Remember). Real preconditions (cooldowns,
    /// inventory, LOS, memory presence, …) land alongside each domain's
    /// compiler-registered cascade handlers in later plans.
    ///
    /// `state.ability_registry` drives the Cast gate: when non-empty, the
    /// first registered ability is treated as each agent's candidate cast
    /// and routed through `evaluate_cast_gate` with the nearest hostile
    /// within engagement range as the inferred target (same heuristic
    /// `UtilityBackend` uses for Attack). An empty registry falls back to
    /// permissive — matches the pre-registry-move behaviour where no
    /// `CastHandler` was registered.
    ///
    /// Audit fix CRITICAL #2. Registry-on-state since the cast-handler
    /// migration retired the `Arc<AbilityRegistry>` plumbing.
    pub fn mark_domain_hook_micros_allowed(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        // Non-cast domain hooks remain permissive (no gate yet).
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            for k in [
                MicroKind::UseItem,      MicroKind::Harvest,
                MicroKind::PlaceTile,    MicroKind::PlaceVoxel,   MicroKind::HarvestVoxel,
                MicroKind::Converse,     MicroKind::ShareStory,
                MicroKind::Communicate,  MicroKind::Ask,          MicroKind::Remember,
            ] {
                self.micro_kind[slot * n_kinds + k as usize] = true;
            }
        }

        // Cast: pass through `evaluate_cast_gate` when the state-borne
        // registry has at least one ability, falling back to permissive
        // for an empty registry.
        let ability_id = if state.ability_registry.is_empty() {
            None
        } else {
            crate::ability::AbilityId::new(1)
        };
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let cast_offset = slot * n_kinds + MicroKind::Cast as usize;
            let allowed = match ability_id {
                Some(ability) => {
                    match inferred_cast_target(state, id) {
                        Some(target) => evaluate_cast_gate(
                            state, &state.ability_registry, id, ability, target,
                        ),
                        None => false,
                    }
                }
                // Empty registry → permissive.
                None => true,
            };
            self.micro_kind[cast_offset] = allowed;
        }
    }
}

/// Pick the nearest hostile within engagement range as the inferred cast
/// target. Task 138 retired the matching Attack-target heuristic
/// (`nearest_other`); Cast's target inference stays here until the Cast
/// mask gains a `from` clause of its own. Returns `None` when no hostile
/// is in range.
fn inferred_cast_target(state: &SimState, caster: AgentId) -> Option<AgentId> {
    let pos = state.agent_pos(caster)?;
    let ct = state.agent_creature_type(caster)?;
    let spatial = state.spatial();
    let mut best: Option<(AgentId, f32)> = None;
    for other in spatial.within_radius(state, pos, state.config.combat.aggro_range) {
        if other == caster { continue; }
        let op = match state.agent_pos(other) { Some(p) => p, None => continue };
        let oc = match state.agent_creature_type(other) { Some(c) => c, None => continue };
        if !ct.is_hostile_to(oc) { continue; }
        let d = pos.distance(op);
        match best {
            None => best = Some((other, d)),
            Some((_, bd)) if d < bd => best = Some((other, d)),
            _ => {}
        }
    }
    best.map(|(id, _)| id)
}
