// crates/engine/src/mask.rs
use crate::ability::{evaluate_cast_gate, AbilityRegistry};
use crate::generated::mask::mask_attack;
use crate::ids::AgentId;
use crate::state::SimState;

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
    pub fn mark_hold_allowed(&mut self, state: &SimState) {
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let offset = slot * MicroKind::ALL.len() + MicroKind::Hold as usize;
            self.micro_kind[offset] = true;
        }
    }
    pub fn mark_move_allowed_if_others_exist(&mut self, state: &SimState) {
        let n_alive = state.agents_alive().count();
        if n_alive < 2 { return; }
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let offset = slot * MicroKind::ALL.len() + MicroKind::MoveToward as usize;
            self.micro_kind[offset] = true;
        }
    }

    /// Mark `Flee` as allowed for every alive agent that has at least one
    /// other alive agent within `AGGRO_RANGE`. No threat → no flee.
    ///
    /// Uses `state.spatial()` so the threat check is `O(N·k)` (k = candidates
    /// in the 3×3 cell neighbourhood) rather than `O(N²)`.
    pub fn mark_flee_allowed_if_threat_exists(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        let spatial = state.spatial();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let self_pos = match state.agent_pos(id) {
                Some(p) => p,
                None    => continue,
            };
            let has_threat = spatial
                .within_radius(state, self_pos, state.config.combat.aggro_range)
                .into_iter()
                .any(|other| other != id);
            if has_threat {
                let offset = slot * n_kinds + MicroKind::Flee as usize;
                self.micro_kind[offset] = true;
            }
        }
    }

    /// Mark `Eat`, `Drink`, and `Rest` as always-allowed for every alive agent.
    /// MVP: no world-state preconditions (e.g. food availability, rest site) —
    /// those land when the inventory/site systems arrive.
    pub fn mark_needs_allowed(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            self.micro_kind[slot * n_kinds + MicroKind::Eat   as usize] = true;
            self.micro_kind[slot * n_kinds + MicroKind::Drink as usize] = true;
            self.micro_kind[slot * n_kinds + MicroKind::Rest  as usize] = true;
        }
    }

    /// Mark `Attack` as allowed for every alive agent that has at least one
    /// valid target as decided by the compiler-emitted `mask_attack`
    /// predicate (see `assets/sim/masks.sim`). The predicate checks:
    /// target is alive, target is hostile to self (per
    /// `crate::rules::is_hostile`), and the two agents are within 2.0m.
    ///
    /// The spatial iterator bounds the candidate set; the predicate
    /// decides per-pair whether the bit should be set. Custom per-agent
    /// attack ranges (`set_agent_attack_range`) are honoured by
    /// broadening the spatial iterator — the predicate still enforces
    /// the DSL-declared 2.0m cap, so custom-range setups only matter
    /// once the attack mask grows per-agent range support in the DSL.
    pub fn mark_attack_allowed_if_target_in_range(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        let spatial = state.spatial();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let self_pos = match state.agent_pos(id) {
                Some(p) => p,
                None    => continue,
            };
            let floor = state.config.combat.attack_range;
            let range = state
                .agent_attack_range(id)
                .unwrap_or(floor)
                .max(floor);
            let has_target = spatial
                .within_radius(state, self_pos, range)
                .into_iter()
                .any(|other| other != id && mask_attack(state, id, other));
            if has_target {
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
    /// `cast_registry`, when provided, overrides the permissive default for
    /// `MicroKind::Cast` with the full `evaluate_cast_gate` predicate: the
    /// first ability in the registry is treated as each agent's candidate
    /// cast, with the nearest hostile within engagement range as the
    /// inferred target (same heuristic `UtilityBackend` uses for Attack).
    /// When the registry is empty or no cast handler is registered,
    /// `MicroKind::Cast` falls back to permissive — matches the pre-gate
    /// behaviour and the legacy `mark_domain_hook_micros_allowed` tests.
    ///
    /// Audit fix CRITICAL #2.
    pub fn mark_domain_hook_micros_allowed(
        &mut self,
        state:         &SimState,
        cast_registry: Option<&AbilityRegistry>,
    ) {
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

        // Cast: pass through `evaluate_cast_gate` when a registry is bound,
        // falling back to permissive when no registry is available.
        let ability_id = cast_registry.and_then(first_registered_ability);
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let cast_offset = slot * n_kinds + MicroKind::Cast as usize;
            let allowed = match (cast_registry, ability_id) {
                (Some(reg), Some(ability)) => {
                    match inferred_cast_target(state, id) {
                        Some(target) => evaluate_cast_gate(state, reg, id, ability, target),
                        None         => false,
                    }
                }
                // No ability registered / no registry bound → permissive.
                _ => true,
            };
            self.micro_kind[cast_offset] = allowed;
        }
    }
}

/// Return the first `AbilityId` (slot 0) in a non-empty registry.
/// Mask-build treats it as each agent's representative cast for the
/// per-agent gate check.
fn first_registered_ability(reg: &AbilityRegistry) -> Option<crate::ability::AbilityId> {
    if reg.is_empty() { return None; }
    crate::ability::AbilityId::new(1)
}

/// Pick the nearest hostile within engagement range as the inferred cast
/// target. Matches the `UtilityBackend` heuristic for Attack: locality +
/// hostility. Returns `None` when no hostile is in range.
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
