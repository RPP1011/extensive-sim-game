// crates/engine/src/mask.rs
//
// Storage primitives for the action mask: MicroKind enum, TargetMask, and
// MaskBuffer. Rule-aware mask-build methods (mark_hold_allowed,
// mark_move_allowed_from_candidates, mark_attack_allowed_from_candidates,
// mark_flee_allowed_if_threat_exists, mark_needs_allowed,
// mark_domain_hook_micros_allowed) and the inferred_cast_target helper are
// DELETED in Plan B1' Task 11 — they called into generated mask functions and
// belong to the rule layer. `engine_rules::step` (emitted in Task 11) will
// own those calls. Only storage primitives remain here.
//
// Interpreter dispatch helpers live in `mod interp` below (feature-gated on
// `interpreted-rules`). They are called from `engine_rules::mask_fill` (the
// generated fill_all function) when the feature is on.
use crate::ids::AgentId;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Interpreted-rules dispatch helpers (feature = "interpreted-rules")
// ---------------------------------------------------------------------------

#[cfg(feature = "interpreted-rules")]
pub mod interp {
    use crate::ids::AgentId;
    use crate::mask::{MicroKind, TargetMask};
    use crate::state::SimState;
    use dsl_ast::eval::mask::LocalParam;
    use dsl_ast::ir::MaskIR;

    /// Lazily parse and cache the DSL compilation from `assets/sim/*.sim`.
    ///
    /// Uses `std::sync::OnceLock` so the parse happens once per test-binary
    /// run. Path is resolved relative to `CARGO_MANIFEST_DIR` so the test
    /// binary can find the asset files regardless of the working directory.
    pub fn compilation() -> &'static dsl_ast::Compilation {
        static COMP: std::sync::OnceLock<dsl_ast::Compilation> = std::sync::OnceLock::new();
        COMP.get_or_init(|| {
            // Path: crates/engine/  →  repo root  →  assets/sim/
            let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
            let read = |name: &str| {
                let path = format!("{root}/assets/sim/{name}");
                std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("interpreted-rules: failed to read {path}: {e}"))
            };
            let events        = read("events.sim");
            let entities      = read("entities.sim");
            let enums         = read("enums.sim");
            let views         = read("views.sim");
            let masks         = read("masks.sim");
            let scoring       = read("scoring.sim");
            let physics       = read("physics.sim");
            let config        = read("config.sim");
            let full = format!(
                "{events}\n{entities}\n{enums}\n{views}\n{masks}\n{scoring}\n{physics}\n{config}"
            );
            dsl_ast::compile(&full)
                .unwrap_or_else(|e| panic!("interpreted-rules: DSL compile error: {e}"))
        })
    }

    /// Build and cache a `Config` from the parsed DSL Compilation.
    ///
    /// Walking `compilation().configs` and extracting `ConfigDefault` literals
    /// means every `config.*.*` read in interpreted mode reflects the current
    /// `assets/sim/config.sim` — without requiring a `cargo build` or a
    /// `state.config` mutation.
    ///
    /// Unknown block / field names are silently skipped; minor DSL/struct
    /// drift will not panic.
    pub fn interp_config() -> &'static engine_data::config::Config {
        use dsl_ast::ast::ConfigDefault;
        use engine_data::config::{
            BeliefConfig, CombatConfig, CommunicationConfig, Config, MovementConfig, NeedsConfig,
        };

        static CFG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
        CFG.get_or_init(|| {
            let comp = compilation();
            let mut combat        = CombatConfig::default();
            let mut movement      = MovementConfig::default();
            let mut needs         = NeedsConfig::default();
            let mut communication = CommunicationConfig::default();

            for block in &comp.configs {
                match block.name.as_str() {
                    "combat" => {
                        for f in &block.fields {
                            match f.name.as_str() {
                                "attack_damage"          => { if let ConfigDefault::Float(v) = f.default { combat.attack_damage = v as f32; } }
                                "attack_range"           => { if let ConfigDefault::Float(v) = f.default { combat.attack_range = v as f32; } }
                                "aggro_range"            => { if let ConfigDefault::Float(v) = f.default { combat.aggro_range = v as f32; } }
                                "engagement_range"       => { if let ConfigDefault::Float(v) = f.default { combat.engagement_range = v as f32; } }
                                "engagement_slow_factor" => { if let ConfigDefault::Float(v) = f.default { combat.engagement_slow_factor = v as f32; } }
                                "kin_flee_bias"          => { if let ConfigDefault::Float(v) = f.default { combat.kin_flee_bias = v as f32; } }
                                "kin_flee_radius"        => { if let ConfigDefault::Float(v) = f.default { combat.kin_flee_radius = v as f32; } }
                                _ => { eprintln!("interpreted-rules: unknown combat config field `{}`", f.name); }
                            }
                        }
                    }
                    "movement" => {
                        for f in &block.fields {
                            match f.name.as_str() {
                                "move_speed_mps"  => { if let ConfigDefault::Float(v) = f.default { movement.move_speed_mps = v as f32; } }
                                "max_move_radius" => { if let ConfigDefault::Float(v) = f.default { movement.max_move_radius = v as f32; } }
                                _ => { eprintln!("interpreted-rules: unknown movement config field `{}`", f.name); }
                            }
                        }
                    }
                    "needs" => {
                        for f in &block.fields {
                            match f.name.as_str() {
                                "eat_restore"   => { if let ConfigDefault::Float(v) = f.default { needs.eat_restore = v as f32; } }
                                "drink_restore" => { if let ConfigDefault::Float(v) = f.default { needs.drink_restore = v as f32; } }
                                "rest_restore"  => { if let ConfigDefault::Float(v) = f.default { needs.rest_restore = v as f32; } }
                                _ => { eprintln!("interpreted-rules: unknown needs config field `{}`", f.name); }
                            }
                        }
                    }
                    "communication" => {
                        for f in &block.fields {
                            match f.name.as_str() {
                                "max_announce_recipients"  => { if let ConfigDefault::Uint(v)  = f.default { communication.max_announce_recipients = v as u32; } }
                                "max_announce_radius"      => { if let ConfigDefault::Float(v) = f.default { communication.max_announce_radius = v as f32; } }
                                "overhear_range"           => { if let ConfigDefault::Float(v) = f.default { communication.overhear_range = v as f32; } }
                                "default_vocal_strength"   => { if let ConfigDefault::Float(v) = f.default { communication.default_vocal_strength = v as f32; } }
                                "channel_speech_range"     => { if let ConfigDefault::Float(v) = f.default { communication.channel_speech_range = v as f32; } }
                                "channel_pack_range"       => { if let ConfigDefault::Float(v) = f.default { communication.channel_pack_range = v as f32; } }
                                "channel_pheromone_range"  => { if let ConfigDefault::Float(v) = f.default { communication.channel_pheromone_range = v as f32; } }
                                "channel_long_range_vocal" => { if let ConfigDefault::Float(v) = f.default { communication.channel_long_range_vocal = v as f32; } }
                                _ => { eprintln!("interpreted-rules: unknown communication config field `{}`", f.name); }
                            }
                        }
                    }
                    _ => {
                        eprintln!("interpreted-rules: unknown config block `{}`", block.name);
                    }
                }
            }

            Config { combat, movement, needs, communication, belief: BeliefConfig::default() }
        })
    }

    /// Look up a `MaskIR` by head name, e.g. `"Hold"`, `"Attack"`.
    pub fn mask_ir(name: &str) -> &'static MaskIR {
        compilation()
            .masks
            .iter()
            .find(|m| m.head.name == name)
            .unwrap_or_else(|| panic!("interpreted-rules: no MaskIR with name `{name}`"))
    }

    /// Walk the self-only mask predicate for all alive agents and set the
    /// corresponding `micro_kind` bit. Mirrors `MaskBuffer::mark_self_predicate`
    /// but routes through the interpreter.
    pub fn mark_self_interp(
        mask_buf: &mut crate::mask::MaskBuffer,
        state: &SimState,
        kind: MicroKind,
        mask_name: &str,
    ) {
        use crate::evaluator::context::EngineReadCtx;
        let ir = mask_ir(mask_name);
        let ctx = EngineReadCtx::new(state);
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let dsl_id = dsl_ast::eval::AgentId::new(id.raw())
                .expect("AgentId raw must be non-zero");
            let allowed = ir.eval(&ctx, dsl_id, &[]);
            let slot = (id.raw() - 1) as usize;
            let offset = slot * n_kinds + kind as usize;
            mask_buf.micro_kind[offset] = allowed;
        }
    }

    /// Walk the Attack candidate_source and call the interpreter predicate for
    /// each candidate. Mirrors `mask_attack_candidates` but routes through
    /// the interpreter.
    pub fn mark_attack_candidates_interp(
        state: &SimState,
        self_id: AgentId,
        out: &mut TargetMask,
    ) {
        use crate::evaluator::context::EngineReadCtx;
        let ir = mask_ir("Attack");
        let ctx = EngineReadCtx::new(state);
        let radius = interp_config().combat.attack_range;
        let pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);
        let dsl_self = dsl_ast::eval::AgentId::new(self_id.raw())
            .expect("AgentId raw must be non-zero");
        let spatial = state.spatial();
        for target in spatial.within_radius(state, pos, radius) {
            if target == self_id { continue; }
            let dsl_tgt = match dsl_ast::eval::AgentId::new(target.raw()) {
                Some(id) => id,
                None => continue,
            };
            if ir.eval(&ctx, dsl_self, &[LocalParam::Agent(dsl_tgt)]) {
                out.push(self_id, MicroKind::Attack, target);
            }
        }
    }

    /// Walk the MoveToward candidate_source and call the interpreter predicate
    /// for each candidate. Mirrors `mask_move_toward_candidates` but routes
    /// through the interpreter.
    pub fn mark_move_toward_candidates_interp(
        state: &SimState,
        self_id: AgentId,
        out: &mut TargetMask,
    ) {
        use crate::evaluator::context::EngineReadCtx;
        let ir = mask_ir("MoveToward");
        let ctx = EngineReadCtx::new(state);
        let radius = interp_config().movement.max_move_radius;
        let pos = state.agent_pos(self_id).unwrap_or(glam::Vec3::ZERO);
        let dsl_self = dsl_ast::eval::AgentId::new(self_id.raw())
            .expect("AgentId raw must be non-zero");
        let spatial = state.spatial();
        for target in spatial.within_radius(state, pos, radius) {
            if target == self_id { continue; }
            let dsl_tgt = match dsl_ast::eval::AgentId::new(target.raw()) {
                Some(id) => id,
                None => continue,
            };
            if ir.eval(&ctx, dsl_self, &[LocalParam::Agent(dsl_tgt)]) {
                out.push(self_id, MicroKind::MoveToward, target);
            }
        }
    }

    /// Evaluate the Cast mask for a given caster + ability via the interpreter.
    /// Returns `true` iff the caster passes all caster-side predicates.
    pub fn mask_cast_interp(
        state: &SimState,
        self_id: AgentId,
        ability: crate::ability::AbilityId,
    ) -> bool {
        use crate::evaluator::context::EngineReadCtx;
        let ir = mask_ir("Cast");
        let ctx = EngineReadCtx::new(state);
        let dsl_self = dsl_ast::eval::AgentId::new(self_id.raw())
            .expect("AgentId raw must be non-zero");
        let dsl_ab = dsl_ast::eval::AbilityId::new(ability.raw())
            .expect("AbilityId raw must be non-zero");
        ir.eval(&ctx, dsl_self, &[LocalParam::Ability(dsl_ab)])
    }
}

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

    /// Set the mask bit for `(agent_slot, kind)` directly. Used by
    /// `engine_rules` mask-build code (Task 11) and low-level tests that
    /// need to write a specific bit without the rule-aware mark_* helpers
    /// (which are deleted — they called generated mask fns from engine_rules).
    pub fn set(&mut self, agent_slot: usize, kind: MicroKind, value: bool) {
        let offset = agent_slot * MicroKind::ALL.len() + kind as usize;
        if let Some(b) = self.micro_kind.get_mut(offset) {
            *b = value;
        }
    }

    /// Read the mask bit for `(agent_slot, kind)`.
    pub fn get(&self, agent_slot: usize, kind: MicroKind) -> bool {
        let offset = agent_slot * MicroKind::ALL.len() + kind as usize;
        self.micro_kind.get(offset).copied().unwrap_or(false)
    }

    /// Number of agents this buffer was sized for.
    pub fn n_agents(&self) -> u32 {
        (self.micro_kind.len() / MicroKind::ALL.len()) as u32
    }

    /// Number of micro-kind slots per agent.
    pub fn n_kinds(&self) -> u32 {
        MicroKind::ALL.len() as u32
    }

    /// Raw read-only view of the micro-kind bit matrix
    /// (length = n_agents × n_kinds, row-major by agent).
    pub fn bits(&self) -> &[bool] {
        &self.micro_kind
    }
}
