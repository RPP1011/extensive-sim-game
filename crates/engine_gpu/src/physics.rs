//! Phase 6e — GPU physics kernel (Piece 2 of the cascade megakernel).
//!
//! This module wires task 187's `emit_physics_wgsl` output into a
//! runnable compute kernel. For each event in an input batch, the
//! shader dispatches every applicable physics rule (via the emitter's
//! `physics_dispatch`), which may write to agent state / views and emit
//! new events into `event_ring`.
//!
//! ## Scope
//!
//! Piece 2 covers ONE cascade iteration: take a batch of events,
//! dispatch physics on each, return the new events + mutated state.
//! The cascade loop (Piece 3) drives this kernel in a fixed-point loop.
//! This module owns the compute pipeline + bind groups; the cascade
//! driver coordinates re-dispatches, view folds, and event drains.
//!
//! ## Rule support matrix (23 rules in `assets/sim/physics.sim`)
//!
//!   | Rule                       | Status    | Notes                        |
//!   |----------------------------|-----------|------------------------------|
//!   | damage                     | FULL      | shield+hp+kill+emit          |
//!   | heal                       | FULL      | hp clamp                     |
//!   | shield                     | FULL      | additive                     |
//!   | stun                       | FULL      | longest-wins                 |
//!   | slow                       | FULL      | longer-or-stronger wins      |
//!   | opportunity_attack         | FULL      | damage + emit                |
//!   | engagement_on_move         | FULL      | spatial + eng + emit         |
//!   | engagement_on_death        | FULL      | engagement teardown          |
//!   | fear_spread_on_death       | FULL      | kin iter + emit              |
//!   | pack_focus_on_engagement   | FULL      | kin iter + emit              |
//!   | rally_on_wound             | FULL      | hp_pct + kin iter + emit     |
//!   | chronicle_death            | STUB      | ChronicleEntry no-op         |
//!   | chronicle_attack           | STUB      | ChronicleEntry no-op         |
//!   | chronicle_engagement       | STUB      | ChronicleEntry no-op         |
//!   | chronicle_wound            | STUB      | ChronicleEntry no-op         |
//!   | chronicle_break            | STUB      | ChronicleEntry no-op         |
//!   | chronicle_rout             | STUB      | ChronicleEntry no-op         |
//!   | chronicle_flee             | STUB      | ChronicleEntry no-op         |
//!   | chronicle_rally            | STUB      | ChronicleEntry no-op         |
//!   | transfer_gold              | STUB      | no gold SoA on GPU           |
//!   | modify_standing            | STUB      | no standing matrix on GPU    |
//!   | record_memory              | STUB      | no memory ring on GPU        |
//!   | cast                       | FULL      | ability LUT + effect emit    |
//!
//! Stubs don't panic — they're no-ops that document the missing state.
//! The chronicle rules are safe to stub because `ChronicleEntry` is
//! non-replayable and doesn't feed back into simulation state.
//! Gold/standing/memory mutations are left to the CPU path; callers
//! that need them run the CPU cascade as authoritative and use the GPU
//! kernel for the replayable subset.
//!
//! ## Event layout reconciliation (task 190 Problem 1)
//!
//! Task 187's emitter originally used `e.payload_N` for payload reads;
//! task 188's event_ring uses `payload: array<u32, 8>`. This commit
//! changed the emitter to `e.payload[N]` so the two WGSL layouts
//! match — one `EventRecord` struct serves both sides.
//!
//! `gpu_emit_event` in event_ring takes 8 payload slots; the emitter
//! now emits 8-arg calls, padding with `0u` for shorter events.
//!
//! ## Agent SoA mutator semantics (task 190 Problem 2)
//!
//! Physics writes to agent state via stub fns (`state_set_agent_hp`,
//! `state_kill_agent`, …). We back those with a packed `AgentSlot`
//! struct buffer bound as `read_write`. Each mutator is a single-word
//! store into one slot's field. `state_kill_agent(id)` sets
//! `alive = 0u`; the CPU-side readback translates that back into
//! `SimState.hot_alive`.
//!
//! ## Ability registry (task 190 Problem 3)
//!
//! Uploaded as three flat u32 arrays:
//!   * `abilities_known[ab]` — 1u if known, else 0u.
//!   * `abilities_cooldown[ab]` — cooldown in ticks.
//!   * `abilities_effects_count[ab]` — effects in the program.
//! Plus `abilities_effects[ab * MAX_EFFECTS + idx]` of packed EffectOp
//! (kind discriminant + two u32 payload words per op).
//!
//! ## Spatial queries (task 190 Problem 5)
//!
//! `query.nearest_hostile_to` and `query.nearby_kin` require pre-
//! computed per-agent results — the physics kernel can't rebuild the
//! spatial hash in-shader without a separate dispatch. The driver
//! pattern: caller runs `GpuSpatialHash::rebuild_and_query` first and
//! seeds the results into `physics`'s spatial-results buffers. If a
//! physics rule fires an event whose handler needs a spatial query on
//! an id the precompute didn't cover, the query returns empty/sentinel
//! (documented behaviour, matches the CPU spatial hash's "unknown
//! agent" fallback).

#![cfg(feature = "gpu")]

use std::fmt;

use bytemuck::{Pod, Zeroable};
use dsl_compiler::emit_physics_wgsl::{emit_physics_dispatcher_wgsl, emit_physics_wgsl, EmitContext};
use dsl_compiler::ir::PhysicsIR;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::SimState;
use crate::event_ring::{
    chronicle_wgsl_prefix, pack_event, unpack_record, wgsl_prefix, DrainOutcome, EventRecord,
    GpuChronicleRing, GpuEventRing, CHRONICLE_RING_WGSL, DEFAULT_CHRONICLE_CAPACITY,
    EVENT_RING_WGSL, PAYLOAD_WORDS,
};

/// Workgroup size for the physics dispatcher.
pub const PHYSICS_WORKGROUP_SIZE: u32 = 64;

/// Binding number of the shared `SimCfg` storage buffer (Task 2.8 of
/// the GPU sim-state refactor). Sits past the resident-only bindings
/// (10 = cfg, 13/14/15 = indirect_args / num_events_buf / resident_cfg)
/// so extending the BGL doesn't disturb the existing slot numbering.
///
/// Both `cs_physics` (sync) and `cs_physics_resident` entry points
/// reference this binding, so the sync + resident BGLs both include an
/// entry at `SIM_CFG_BINDING`. The sync path binds a pool-owned
/// `sync_sim_cfg_buf` (refreshed per `run_batch` via
/// `SimCfg::from_state(state)`); the resident path binds the caller-
/// supplied batch-scope SimCfg buffer the cascade driver threads in.
pub const SIM_CFG_BINDING: u32 = 16;

/// Max effects per ability program. The ability-registry buffer is
/// flat-indexed as `ab * MAX_EFFECTS + effect_idx`. Bumping this grows
/// the registry buffer size linearly with the ability count. 8 matches
/// the CPU `MAX_EFFECTS_PER_PROGRAM` in `engine::ability`.
pub const MAX_EFFECTS: usize = 8;

/// Max ability id + 1 the physics kernel provisions storage for.
/// Abilities with higher ids fall through to `abilities_is_known(ab) ==
/// false` and the `cast` rule silently skips them, matching the CPU
/// behaviour.
pub const MAX_ABILITIES: usize = 256;

// ---------------------------------------------------------------------------
// GPU-POD wire types
// ---------------------------------------------------------------------------

/// Packed per-slot agent state the physics kernel reads and writes.
/// One struct per agent slot; dead slots have `alive = 0`.
///
/// Field order pins the WGSL `AgentSlot` layout below — any reorder or
/// insert forces a WGSL update in lockstep.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuAgentSlot {
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub attack_damage: f32,
    pub alive: u32,
    pub creature_type: u32,
    /// 1-based raw AgentId of partner, or `0xFFFFFFFF` for "none".
    pub engaged_with: u32,
    pub stun_expires_at: u32,
    pub slow_expires_at: u32,
    /// i16 promoted to u32 via sign-preserving `as u16 as u32`.
    pub slow_factor_q8: u32,
    pub cooldown_next_ready: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl GpuAgentSlot {
    pub const ENGAGED_NONE: u32 = 0xFFFF_FFFFu32;
}

/// Packed ability-effect op. WGSL `EffectOp { kind: u32, p0: u32, p1: u32 }`
/// with a trailing pad to keep the stride at 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq, Eq)]
pub struct GpuEffectOp {
    pub kind: u32,
    pub p0: u32,
    pub p1: u32,
    pub _pad: u32,
}

impl GpuEffectOp {
    /// Pack a CPU `engine::ability::EffectOp` into its GPU wire form.
    ///
    /// Discriminants match the `EFFECT_OP_KIND_*` constants emitted into
    /// the physics shader (see `build_physics_shader_with_chronicle`).
    /// Payload words are laid out per-variant:
    ///
    /// * `Damage / Heal / Shield` — `p0` = f32 amount bits, `p1` = 0.
    /// * `Stun` — `p0` = duration_ticks, `p1` = 0.
    /// * `Slow` — `p0` = duration_ticks, `p1` = i16 factor_q8 zero-extended.
    /// * `TransferGold` — `p0` = i32 amount bit pattern, `p1` = 0.
    /// * `ModifyStanding` — `p0` = i16 delta bit pattern, `p1` = 0.
    /// * `CastAbility` — `p0` = AbilityId raw, `p1` = TargetSelector disc.
    ///
    /// Reorganising the layout requires lockstep updates to the
    /// `EFFECT_OP_KIND_*` consts and the dispatcher's `match op` arms.
    pub fn from_effect_op(op: &engine::ability::EffectOp) -> Self {
        use engine::ability::EffectOp;
        match *op {
            EffectOp::Damage { amount } => GpuEffectOp {
                kind: 0,
                p0: amount.to_bits(),
                p1: 0,
                _pad: 0,
            },
            EffectOp::Heal { amount } => GpuEffectOp {
                kind: 1,
                p0: amount.to_bits(),
                p1: 0,
                _pad: 0,
            },
            EffectOp::Shield { amount } => GpuEffectOp {
                kind: 2,
                p0: amount.to_bits(),
                p1: 0,
                _pad: 0,
            },
            EffectOp::Stun { duration_ticks } => GpuEffectOp {
                kind: 3,
                p0: duration_ticks,
                p1: 0,
                _pad: 0,
            },
            EffectOp::Slow { duration_ticks, factor_q8 } => GpuEffectOp {
                kind: 4,
                p0: duration_ticks,
                p1: (factor_q8 as u16) as u32,
                _pad: 0,
            },
            EffectOp::TransferGold { amount } => GpuEffectOp {
                kind: 5,
                p0: amount as u32,
                p1: 0,
                _pad: 0,
            },
            EffectOp::ModifyStanding { delta } => GpuEffectOp {
                kind: 6,
                p0: (delta as u16) as u32,
                p1: 0,
                _pad: 0,
            },
            EffectOp::CastAbility { ability, selector } => GpuEffectOp {
                kind: 7,
                p0: ability.raw(),
                p1: selector as u32,
                _pad: 0,
            },
        }
    }
}

/// Pre-computed per-slot kin list — matches `GpuQueryResult`'s ids
/// layout but kept as a separate struct so the physics kernel can be
/// loaded without depending on spatial_gpu's types verbatim. We wrap
/// spatial_gpu's readback at the driver level.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuKinList {
    pub count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub ids: [u32; crate::spatial_gpu::K as usize],
}

impl Default for GpuKinList {
    fn default() -> Self {
        Self {
            count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            ids: [u32::MAX; crate::spatial_gpu::K as usize],
        }
    }
}

/// Kernel-local config uniform. 32 bytes — preserved across Task 2.8
/// to avoid perturbing the uniform's 16-byte WGSL alignment.
///
/// Task 2.8 of the GPU sim-state refactor migrated the world-scalar
/// fields (`tick`, `combat_engagement_range`, `cascade_max_iterations`)
/// onto the shared `SimCfg` storage buffer bound at `SIM_CFG_BINDING`.
/// The remaining fields are all kernel-local: `num_events` varies per
/// sync dispatch; `agent_cap`, `max_abilities`, `max_effects` are
/// build-time bounds that the kernel uses for loop caps and slot
/// validation.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct PhysicsCfg {
    pub num_events: u32,
    pub agent_cap: u32,
    pub max_abilities: u32,
    pub max_effects: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Task B7 — resident-path uniform. Tells the resident kernel which
/// slot to read its `num_events` from (`read_slot`) and which slot to
/// publish the next iteration's workgroup count + event count to
/// (`write_slot`). Agent cap + other global scalars keep coming from
/// the shared `PhysicsCfg` so the resident path reuses the existing
/// uniform rather than duplicating it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct ResidentPhysicsCfg {
    pub read_slot: u32,
    pub write_slot: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

/// Physics-kernel error surface.
#[derive(Debug)]
pub enum PhysicsError {
    EmitWgsl(String),
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::EmitWgsl(s) => write!(f, "physics emit WGSL: {s}"),
            PhysicsError::ShaderCompile(s) => write!(f, "physics shader compile: {s}"),
            PhysicsError::Dispatch(s) => write!(f, "physics dispatch: {s}"),
        }
    }
}

impl std::error::Error for PhysicsError {}

// ---------------------------------------------------------------------------
// Agent SoA pack / unpack
// ---------------------------------------------------------------------------

/// Read the SimState's hot agent arrays into `Vec<GpuAgentSlot>` —
/// one slot per agent_cap.
pub fn pack_agent_slots(state: &SimState) -> Vec<GpuAgentSlot> {
    let cap = state.agent_cap() as usize;
    let mut out = Vec::with_capacity(cap);
    for slot in 0..cap {
        let id = match AgentId::new(slot as u32 + 1) {
            Some(id) => id,
            None => {
                out.push(GpuAgentSlot::zeroed());
                continue;
            }
        };
        let alive = state.agent_alive(id);
        if !alive {
            let mut z = GpuAgentSlot::zeroed();
            z.alive = 0;
            z.engaged_with = GpuAgentSlot::ENGAGED_NONE;
            out.push(z);
            continue;
        }
        let pos = state.agent_pos(id).unwrap_or(glam::Vec3::ZERO);
        let engaged = state
            .agent_engaged_with(id)
            .map(|p| p.raw())
            .unwrap_or(GpuAgentSlot::ENGAGED_NONE);
        let slow_factor = state.agent_slow_factor_q8(id).unwrap_or(0);
        // `slow_factor_q8` is an i16; encode as u16 bits then widen so
        // the round-trip preserves negative values (kernel also reads it
        // as a u32 with sign-extension where needed).
        let slow_factor_u32 = (slow_factor as u16) as u32;
        out.push(GpuAgentSlot {
            hp: state.agent_hp(id).unwrap_or(0.0),
            max_hp: state.agent_max_hp(id).unwrap_or(0.0),
            shield_hp: state.agent_shield_hp(id).unwrap_or(0.0),
            attack_damage: state.agent_attack_damage(id).unwrap_or(0.0),
            alive: 1,
            creature_type: state
                .agent_creature_type(id)
                .map(|c| c as u32)
                .unwrap_or(u32::MAX),
            engaged_with: engaged,
            stun_expires_at: state.agent_stun_expires_at(id).unwrap_or(0),
            slow_expires_at: state.agent_slow_expires_at(id).unwrap_or(0),
            slow_factor_q8: slow_factor_u32,
            cooldown_next_ready: state.agent_cooldown_next_ready(id).unwrap_or(0),
            pos_x: pos.x,
            pos_y: pos.y,
            pos_z: pos.z,
            _pad0: 0,
            _pad1: 0,
        });
    }
    out
}

/// Apply a readback of `Vec<GpuAgentSlot>` to `SimState` — used by the
/// parity test to materialise the kernel's state writes back onto the
/// authoritative SimState for comparison.
///
/// Field writes always fire (even for newly-killed slots) so the CPU
/// `hot_*` arrays match the GPU's post-kill snapshot. The alive-flag
/// transition is deferred to a `kill_agent` call AFTER the field
/// writes so the kill-teardown (pool drop + spatial remove) runs on
/// the correct field state.
pub fn unpack_agent_slots(state: &mut SimState, slots: &[GpuAgentSlot]) {
    for (slot_idx, s) in slots.iter().enumerate() {
        let id = match AgentId::new(slot_idx as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        let currently_alive = state.agent_alive(id);
        // Skip slots that are dead on both sides — they had no event
        // targeting them so their GPU snapshot is just zeros.
        if !currently_alive && s.alive == 0 {
            continue;
        }
        // Mirror every field. `set_agent_hp(dead)` is a no-op on the
        // CPU side, so we update while the CPU still considers the
        // slot alive, THEN call `kill_agent` to flip the flag.
        state.set_agent_hp(id, s.hp);
        state.set_agent_shield_hp(id, s.shield_hp);
        state.set_agent_stun_expires_at(id, s.stun_expires_at);
        state.set_agent_slow_expires_at(id, s.slow_expires_at);
        let factor_i16 = (s.slow_factor_q8 & 0xFFFF) as u16 as i16;
        state.set_agent_slow_factor_q8(id, factor_i16);
        state.set_agent_cooldown_next_ready(id, s.cooldown_next_ready);
        let engaged = if s.engaged_with == GpuAgentSlot::ENGAGED_NONE {
            None
        } else {
            AgentId::new(s.engaged_with)
        };
        state.set_agent_engaged_with(id, engaged);
        if currently_alive && s.alive == 0 {
            state.kill_agent(id);
        }
    }
}

// ---------------------------------------------------------------------------
// Ability registry pack
// ---------------------------------------------------------------------------

/// Sentinel written into `PackedAbilityRegistry::hints` when the
/// ability has no `hint:` field. Chosen so any legal
/// `AbilityHint::discriminant()` (0..=3 today) stays distinguishable.
///
/// WGSL consumers (Phase 4) will compare against this constant before
/// dispatching a hint-specific branch; `HINT_NONE_SENTINEL` is not a
/// valid discriminant.
pub const HINT_NONE_SENTINEL: u32 = 0xFFu32;

/// Flat ability registry ready for upload.
///
/// The `hints` and `tag_values` fields are the Phase 1 addition of the
/// GPU ability-evaluation subsystem (see
/// `docs/spec/gpu.md (§5)`).
/// They carry the scoring surface that Phase 4 will bind and read; no
/// physics kernel consumes them today. `empty()` initialises both to
/// their "no data" defaults so existing tests that skip
/// `CastAbility` / ability scoring keep passing unchanged.
pub struct PackedAbilityRegistry {
    pub known: Vec<u32>,
    pub cooldown: Vec<u32>,
    pub effects_count: Vec<u32>,
    /// `effects[ab * MAX_EFFECTS + effect_idx]`.
    pub effects: Vec<GpuEffectOp>,
    /// `hints[ab]` — coarse category hint per ability, encoded as
    /// `AbilityHint::discriminant() as u32`. `HINT_NONE_SENTINEL`
    /// marks abilities whose DSL source omits the `hint:` field.
    ///
    /// Unbound in Phase 1 — Phase 4 wires this into `pick_ability.wgsl`.
    pub hints: Vec<u32>,
    /// Per-ability-per-tag power rating. Flat layout
    /// `tag_values[ab * NUM_ABILITY_TAGS + tag_idx]`, stride
    /// `AbilityTag::COUNT`. Missing tags read as `0.0f32`, matching the
    /// DSL contract of `ability::tag(UNKNOWN) == 0`.
    ///
    /// Unbound in Phase 1 — Phase 4 wires this into `pick_ability.wgsl`.
    pub tag_values: Vec<f32>,
}

impl PackedAbilityRegistry {
    /// Empty registry — every ability-id lookup fails.
    /// Used by tests that don't exercise `cast`.
    pub fn empty() -> Self {
        Self {
            known: vec![0u32; MAX_ABILITIES],
            cooldown: vec![0u32; MAX_ABILITIES],
            effects_count: vec![0u32; MAX_ABILITIES],
            effects: vec![GpuEffectOp::default(); MAX_ABILITIES * MAX_EFFECTS],
            hints: vec![HINT_NONE_SENTINEL; MAX_ABILITIES],
            tag_values: vec![0.0f32; MAX_ABILITIES * engine::ability::AbilityTag::COUNT],
        }
    }

    /// Pack a CPU `AbilityRegistry` into upload-ready flat buffers.
    ///
    /// Extracts the full ability surface exposed to GPU scoring:
    /// `known` / `cooldown` / `effects_count` / `effects` (existing
    /// cascade-cast inputs) plus `hints` / `tag_values` (Phase 1
    /// addition for the forthcoming `pick_ability.wgsl`).
    ///
    /// Registries with more than `MAX_ABILITIES` entries are truncated;
    /// overflow is a configuration bug, not a runtime error, and slots
    /// past the budget are dropped. Known programs with more than
    /// `MAX_EFFECTS` effects are also truncated — `effects_count`
    /// reports the truncated count so the kernel's per-ability loop
    /// stays in bounds.
    pub fn pack(registry: &engine::ability::AbilityRegistry) -> Self {
        use engine::ability::{AbilityId, AbilityTag};

        let mut out = Self::empty();
        let n = registry.len().min(MAX_ABILITIES);
        for slot in 0..n {
            // AbilityId is 1-based; slot 0 => id 1.
            let id = match AbilityId::new((slot as u32) + 1) {
                Some(id) => id,
                None => continue,
            };
            let Some(program) = registry.get(id) else { continue };

            out.known[slot] = 1;
            out.cooldown[slot] = program.gate.cooldown_ticks;

            let effect_count = program.effects.len().min(MAX_EFFECTS);
            out.effects_count[slot] = effect_count as u32;
            for (i, op) in program.effects.iter().take(effect_count).enumerate() {
                out.effects[slot * MAX_EFFECTS + i] = GpuEffectOp::from_effect_op(op);
            }

            out.hints[slot] = program
                .hint
                .map(|h| h.discriminant() as u32)
                .unwrap_or(HINT_NONE_SENTINEL);

            let stride = AbilityTag::COUNT;
            for &(tag, value) in program.tags.iter() {
                out.tag_values[slot * stride + tag.index()] = value;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Shader source construction
// ---------------------------------------------------------------------------

/// Build the full WGSL shader that drives physics dispatch for a batch
/// of events. Layout:
///
///   * constants, enum + state-stub prelude
///   * agent SoA bindings (read_write)
///   * ability registry + spatial + config bindings (read)
///   * event ring bindings (read_write) — via EVENT_RING_WGSL
///   * events_in buffer (read)
///   * per-rule emitted fns (via emit_physics_wgsl)
///   * physics_dispatch (via emit_physics_dispatcher_wgsl)
///   * `cs_physics` entry point — one thread per event, dispatches
///     physics on its event slot.
pub fn build_physics_shader(
    physics: &[PhysicsIR],
    ctx: &EmitContext<'_>,
    event_ring_capacity: u32,
) -> Result<String, PhysicsError> {
    build_physics_shader_with_chronicle(
        physics,
        ctx,
        event_ring_capacity,
        DEFAULT_CHRONICLE_CAPACITY,
        false, // sync path: no gold_buf binding, gold stubs stay no-op
        false, // sync path: no standing_storage, adjust_standing stays no-op
        false, // sync path: no memory_storage, push_agent_memory stays no-op
        true,  // both sync + resident bind the alive_bitmap at slot 22
    )
}

/// Task 203 — explicit-capacity variant. `chronicle_ring_capacity`
/// picks the size of the dedicated chronicle ring that physics routes
/// `emit ChronicleEntry` sites into. Exposed separately so tests can
/// shrink the chronicle ring without touching the main event ring
/// configuration.
pub fn build_physics_shader_with_chronicle(
    physics: &[PhysicsIR],
    ctx: &EmitContext<'_>,
    event_ring_capacity: u32,
    chronicle_ring_capacity: u32,
    has_gold_buf: bool,
    has_standing_storage: bool,
    has_memory_storage: bool,
    has_alive_bitmap: bool,
) -> Result<String, PhysicsError> {
    let mut out = String::new();
    out.push_str(&wgsl_prefix(event_ring_capacity));
    out.push_str(&chronicle_wgsl_prefix(chronicle_ring_capacity));
    out.push_str(&format!(
        "const PHYSICS_MAX_EFFECTS: u32 = {}u;\n\
         const PHYSICS_MAX_ABILITIES: u32 = {}u;\n\
         const PHYSICS_SPATIAL_K: u32 = {}u;\n\
         const ENGAGED_SENTINEL: u32 = 0xFFFFFFFFu;\n",
        MAX_EFFECTS,
        MAX_ABILITIES,
        crate::spatial_gpu::K,
    ));

    // ---- Enum discriminants ----
    //
    // The emitter lowers `TargetSelector::Target` to
    // `TARGET_SELECTOR_TARGET`. Every enum the physics rules reference
    // needs a const here. `TargetSelector` is the only one physics.sim
    // currently uses (see `cast`'s `CastAbility` branch).
    out.push_str(
        "const TARGET_SELECTOR_SELF: u32 = 0u;\n\
         const TARGET_SELECTOR_TARGET: u32 = 1u;\n",
    );

    // ---- Event-kind constants the emitter references ----
    //
    // The emitter references `EVENT_KIND_<SCREAMING_SNAKE>` for every
    // event it destructures or emits; event_ring only defines the
    // numeric tags as Rust constants. We mirror them in WGSL here.
    out.push_str(&event_kind_consts());

    // ---- EffectOp discriminants (match cast's `match op { ... }`) ----
    //
    // Numeric values match `crates/engine_rules/src/enums/effect_op.rs`
    // ordinals. If the DSL enum order changes these must move in
    // lockstep.
    out.push_str(
        "const EFFECT_OP_KIND_DAMAGE: u32 = 0u;\n\
         const EFFECT_OP_KIND_HEAL: u32 = 1u;\n\
         const EFFECT_OP_KIND_SHIELD: u32 = 2u;\n\
         const EFFECT_OP_KIND_STUN: u32 = 3u;\n\
         const EFFECT_OP_KIND_SLOW: u32 = 4u;\n\
         const EFFECT_OP_KIND_TRANSFER_GOLD: u32 = 5u;\n\
         const EFFECT_OP_KIND_MODIFY_STANDING: u32 = 6u;\n\
         const EFFECT_OP_KIND_CAST_ABILITY: u32 = 7u;\n",
    );

    // ---- Struct definitions ----
    out.push_str(
        "\nstruct AgentSlot {\n\
         \x20 hp: f32,\n\
         \x20 max_hp: f32,\n\
         \x20 shield_hp: f32,\n\
         \x20 attack_damage: f32,\n\
         \x20 alive: u32,\n\
         \x20 creature_type: u32,\n\
         \x20 engaged_with: u32,\n\
         \x20 stun_expires_at: u32,\n\
         \x20 slow_expires_at: u32,\n\
         \x20 slow_factor_q8: u32,\n\
         \x20 cooldown_next_ready: u32,\n\
         \x20 pos_x: f32,\n\
         \x20 pos_y: f32,\n\
         \x20 pos_z: f32,\n\
         \x20 _pad0: u32,\n\
         \x20 _pad1: u32,\n\
         };\n\n\
         struct EffectOp {\n\
         \x20 kind: u32,\n\
         \x20 p0: u32,\n\
         \x20 p1: u32,\n\
         \x20 _pad: u32,\n\
         };\n\n\
         struct KinList {\n\
         \x20 count: u32,\n\
         \x20 _pad0: u32,\n\
         \x20 _pad1: u32,\n\
         \x20 _pad2: u32,\n\
         \x20 ids: array<u32, 32>,\n\
         };\n\n\
         struct PhysicsConfig {\n\
         \x20 num_events: u32,\n\
         \x20 agent_cap: u32,\n\
         \x20 max_abilities: u32,\n\
         \x20 max_effects: u32,\n\
         \x20 _pad0: u32,\n\
         \x20 _pad1: u32,\n\
         \x20 _pad2: u32,\n\
         \x20 _pad3: u32,\n\
         };\n\n",
    );

    // ---- Bindings ----
    //
    // All on @group(0) to keep the driver simple. 16 bindings max: we
    // pack agent state into a single `AgentSlot` struct buffer so we
    // don't burn 10+ bindings on per-field arrays.
    //
    //   0: agents          (read_write, storage)  AgentSlot array
    //   1: abilities_known (read, storage)        u32 array
    //   2: abilities_cooldown (read, storage)     u32 array
    //   3: abilities_effects_count (read, storage) u32 array
    //   4: abilities_effects (read, storage)      EffectOp array
    //   5: kin_lists       (read, storage)        KinList array (per agent slot)
    //   6: nearest_hostile (read, storage)        u32 array (per agent slot)
    //   7: events_in       (read, storage)        EventRecord array
    //   8: event_ring      (read_write, storage)  EventRecord array
    //   9: event_ring_tail (read_write, atomic)   atomic<u32>
    //   10: cfg            (uniform)              PhysicsConfig
    // Bindings (minus event_ring + event_ring_tail, which come with
    // EVENT_RING_WGSL below).
    out.push_str(
        "@group(0) @binding(0)  var<storage, read_write> agents: array<AgentSlot>;\n\
         @group(0) @binding(1)  var<storage, read>       ab_known_buf: array<u32>;\n\
         @group(0) @binding(2)  var<storage, read>       ab_cooldown_buf: array<u32>;\n\
         @group(0) @binding(3)  var<storage, read>       ab_effects_count_buf: array<u32>;\n\
         @group(0) @binding(4)  var<storage, read>       ab_effects_buf: array<EffectOp>;\n\
         @group(0) @binding(5)  var<storage, read>       kin_lists: array<KinList>;\n\
         @group(0) @binding(6)  var<storage, read>       nearest_hostile_buf: array<u32>;\n\
         @group(0) @binding(8)  var<storage, read_write> event_ring: array<EventRecord>;\n\
         @group(0) @binding(9)  var<storage, read_write> event_ring_tail: atomic<u32>;\n\
         @group(0) @binding(10) var<uniform>             cfg: PhysicsConfig;\n\n",
    );

    // SimCfg — shared world-scalar storage buffer (Task 2.8 of the GPU
    // sim-state refactor). Declared via the dsl_compiler helper so the
    // struct layout stays in lockstep with `engine_gpu::sim_cfg::SimCfg`
    // — any future field added there becomes visible to physics
    // without a round of edits here. The physics emitter rewrites
    // `config.combat.engagement_range` → `sim_cfg.engagement_range`,
    // `cascade.max_iterations` → `sim_cfg.cascade_max_iterations`, and
    // `state.tick` → `wgsl_world_tick` (function-scope alias to
    // `sim_cfg.tick` injected by `wrap_rule_with_tick_alias`).
    dsl_compiler::emit_sim_cfg::emit_sim_cfg_struct_wgsl(&mut out, SIM_CFG_BINDING);

    // Pull in event_ring's WGSL: defines `EventRecord` struct and the
    // `gpu_emit_event(...)` + per-kind helpers the physics emitter's
    // output calls into.
    out.push_str(EVENT_RING_WGSL);
    out.push_str("\n");

    // events_in binding comes AFTER EVENT_RING_WGSL because it refers
    // to `EventRecord`, which that module defines.
    out.push_str(
        "@group(0) @binding(7) var<storage, read> events_in: array<EventRecord>;\n\n",
    );

    // Task 203 — chronicle ring. Parallel to the main event ring but
    // only receives `emit ChronicleEntry` records. Physics bodies route
    // chronicle emits through `gpu_emit_chronicle_event(...)` (from
    // CHRONICLE_RING_WGSL below); the main ring sees zero chronicle
    // traffic, shrinking its atomic-tail contention + drain cost.
    //
    // Bindings 11 / 12 sit after the cfg uniform (binding 10) so
    // adding them doesn't disturb the preexisting bind-group ordering.
    out.push_str(
        "@group(0) @binding(11) var<storage, read_write> chronicle_ring: array<ChronicleRecord>;\n\
         @group(0) @binding(12) var<storage, read_write> chronicle_ring_tail: atomic<u32>;\n",
    );
    out.push_str(CHRONICLE_RING_WGSL);
    out.push_str("\n");

    // Phase 3 Task 3.4 — gold ledger side buffer (resident path only).
    // Declared before `state_stub_fns` emits its atomic bodies so the
    // `gold_buf` identifier is in scope when the stubs reference it.
    // Sync path skips this declaration and emits no-op stub bodies,
    // keeping the sync shader's binding set unchanged (0..12 + 16).
    if has_gold_buf {
        out.push_str(
            "\n// ---- Phase 3 Task 3.4 — gold ledger side buffer ----\n\
             @group(0) @binding(17) var<storage, read_write> gold_buf: array<atomic<i32>>;\n\n",
        );
    }

    // Task #79 SP-4 — standing view storage (resident path only).
    // Bindings 18 (records) + 19 (counts). Layout matches the WGSL
    // fold kernel's struct: `{ other: u32, value: i32, anchor_tick:
    // u32, _pad: u32 }` = 16 B. K = 8. Declared before
    // `state_stub_fns` so the real `state_adjust_standing` body has
    // the symbols in scope. Sync path skips: its stub stays no-op.
    if has_standing_storage {
        out.push_str(
            "\n// ---- Task #79 SP-4 — standing view storage ----\n\
             const STANDING_K: u32 = 8u;\n\
             const STANDING_CLAMP_POS: i32 = 1000;\n\
             const STANDING_CLAMP_NEG: i32 = -1000;\n\
             struct StandingEdgeGpu {\n\
             \x20 other:       u32,\n\
             \x20 value:       i32,\n\
             \x20 anchor_tick: u32,\n\
             \x20 _pad:        u32,\n\
             };\n\
             @group(0) @binding(18) var<storage, read_write> standing_records_buf: array<StandingEdgeGpu>;\n\
             @group(0) @binding(19) var<storage, read_write> standing_counts_buf:  array<atomic<u32>>;\n\n",
        );
    }

    // Subsystem 2 Phase 4 PR-4 — memory view storage (resident path
    // only). Bindings 20 (records) + 21 (cursors). Struct matches the
    // 24-byte `MemoryEventGpu` documented in
    // `crates/engine_gpu/src/view_storage_per_entity_ring.rs`:
    // `source | kind | payload_lo | payload_hi | confidence | tick`.
    // K = 64. Declared before `state_stub_fns` so the real
    // `state_push_agent_memory` body has the symbols in scope.
    // Sync path skips: its stub stays no-op (CPU `cold_state_replay`
    // still handles `RecordMemory` events there).
    if has_memory_storage {
        out.push_str(
            "\n// ---- Subsystem 2 Phase 4 PR-4 — memory view storage ----\n\
             const MEMORY_K: u32 = 64u;\n\
             struct MemoryEventGpu {\n\
             \x20 source:      u32,\n\
             \x20 kind:        u32,\n\
             \x20 payload_lo:  u32,\n\
             \x20 payload_hi:  u32,\n\
             \x20 confidence:  u32,\n\
             \x20 tick:        u32,\n\
             };\n\
             @group(0) @binding(20) var<storage, read_write> memory_records_buf: array<MemoryEventGpu>;\n\
             @group(0) @binding(21) var<storage, read_write> memory_cursors_buf: array<atomic<u32>>;\n\n",
        );
    }

    // ---- Alive bitmap (binding 22) ----
    //
    // Packed `array<u32>`, one bit per agent slot, written once per
    // tick by `alive_pack_kernel` (resident path) or host-side
    // packer (sync path). Every `agents.alive(x)` predicate in
    // physics rule guards lowers to `alive_bit(slot_of(x))` —
    // avoiding a 64-byte `AgentSlot` cacheline read.
    if has_alive_bitmap {
        out.push_str(
            "\n// ---- Alive bitmap — per-tick packed alive array ----\n\
             @group(0) @binding(22) var<storage, read> alive_bitmap: array<u32>;\n\
             fn alive_bit(slot: u32) -> bool {\n\
             \x20   return ((alive_bitmap[slot >> 5u] >> (slot & 31u)) & 1u) != 0u;\n\
             }\n\n",
        );
    }

    // ---- State stub fns ----
    //
    // Every stub writes/reads exactly one field of the `agents` buffer.
    // Stubs that the CPU side doesn't have a GPU-side source for (gold,
    // standing, memory) are no-ops documented as such.
    out.push_str(&state_stub_fns(
        has_gold_buf,
        has_standing_storage,
        has_memory_storage,
    ));

    // ---- `cfg` namespace-field lookups ----
    //
    // Task 2.8 of the GPU sim-state refactor migrated the emitter's
    // world-scalar references off the per-kernel `cfg` uniform onto
    // the shared `SimCfg` storage buffer. The emitter now lowers:
    //
    //   * `config.combat.engagement_range` → `sim_cfg.engagement_range`
    //   * `cascade.max_iterations`         → `sim_cfg.cascade_max_iterations`
    //
    // Those fields are read directly from the SimCfg binding declared
    // above. Kernel-local fields (`num_events`, `agent_cap`,
    // `max_abilities`, `max_effects`) keep living on `cfg`.

    // ---- `wgsl_world_tick` (tick alias) ----
    //
    // The emitter spells every `state.tick` reference as the bare
    // identifier `wgsl_world_tick` so integration decides where the
    // value comes from. Post-Task-2.8 it comes from the shared SimCfg
    // storage buffer (`sim_cfg.tick`); the seed-indirect kernel's
    // atomic `tick++` is the sole writer, and every physics read sees
    // the post-increment value. Declaring the alias as a function-
    // scope `let` inside each rule fn (via `wrap_rule_with_tick_alias`)
    // keeps the emitter output unchanged.

    // ---- Spatial query wrappers ----
    //
    // The emitter calls `spatial_nearest_hostile_to(agent, radius)` and
    // `spatial_nearby_kin_count(agent, radius)` +
    // `spatial_nearby_kin_at(agent, radius, idx)`. We back these with
    // pre-computed per-slot result buffers. The radius argument is
    // ignored (the driver pre-computed with a fixed 12 m kin radius +
    // the config's engagement range for nearest hostile — see the
    // driver docs for how to override).
    out.push_str(&spatial_stub_fns());

    // ---- Abilities registry stubs ----
    out.push_str(
        "fn abilities_is_known(ab: u32) -> bool {\n\
         \x20   if (ab >= cfg.max_abilities) { return false; }\n\
         \x20   return ab_known_buf[ab] != 0u;\n\
         }\n\
         fn abilities_cooldown_ticks(ab: u32) -> u32 {\n\
         \x20   if (ab >= cfg.max_abilities) { return 0u; }\n\
         \x20   return ab_cooldown_buf[ab];\n\
         }\n\
         fn abilities_effects_count(ab: u32) -> u32 {\n\
         \x20   if (ab >= cfg.max_abilities) { return 0u; }\n\
         \x20   return ab_effects_count_buf[ab];\n\
         }\n\
         fn abilities_effect_op_at(ab: u32, idx: u32) -> EffectOp {\n\
         \x20   var z: EffectOp;\n\
         \x20   z.kind = 0u; z.p0 = 0u; z.p1 = 0u; z._pad = 0u;\n\
         \x20   if (ab >= cfg.max_abilities) { return z; }\n\
         \x20   if (idx >= cfg.max_effects) { return z; }\n\
         \x20   return ab_effects_buf[ab * cfg.max_effects + idx];\n\
         }\n\n",
    );

    // ---- Emit every physics rule ----
    for rule in physics {
        match emit_physics_wgsl(rule, ctx) {
            Ok(wgsl) => {
                out.push_str(&wrap_rule_with_tick_alias(&wgsl));
                out.push('\n');
            }
            Err(e) => {
                return Err(PhysicsError::EmitWgsl(format!(
                    "rule `{}`: {}",
                    rule.name, e
                )));
            }
        }
    }

    // ---- Dispatcher ----
    out.push_str(&emit_physics_dispatcher_wgsl(physics, ctx));
    out.push('\n');

    // ---- Entry point ----
    out.push_str(&format!(
        "@compute @workgroup_size({PHYSICS_WORKGROUP_SIZE})\n\
         fn cs_physics(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \x20   let i = gid.x;\n\
         \x20   if (i >= cfg.num_events) {{ return; }}\n\
         \x20   physics_dispatch(i);\n\
         }}\n",
    ));

    Ok(out)
}

/// Task B7 — build the resident-path physics shader.
///
/// Mirrors [`build_physics_shader_with_chronicle`] but emits:
///   * three additional bindings (13/14/15) for the indirect-args
///     cascade: `indirect_args` (storage, read_write,
///     `array<vec3<u32>>`), `num_events_buf` (storage, read_write,
///     `array<u32>`), and `resident_cfg` (uniform,
///     `{read_slot, write_slot, ...}`).
///   * a separate compute entry point `cs_physics_resident` that reads
///     its `num_events` bound from `num_events_buf[resident_cfg.read_slot]`
///     instead of `cfg.num_events`, and — from thread `gid.x == 0u` only
///     — writes the next iteration's indirect args back into slot
///     `resident_cfg.write_slot`.
///
/// Concat-at-call-site vs emitter extension: we build a FULL separate
/// shader here (not a string-concat epilogue over the sync shader
/// source) because the resident path needs a different bind-group
/// layout — appending to the sync shader would force the resident BGL
/// into the sync pipeline and break `run_batch`'s byte-identical
/// behaviour. Sharing the WGSL body text via this dedicated builder
/// keeps both shaders in lockstep without touching `dsl_compiler`.
pub fn build_physics_shader_resident(
    physics: &[PhysicsIR],
    ctx: &EmitContext<'_>,
    event_ring_capacity: u32,
    chronicle_ring_capacity: u32,
) -> Result<String, PhysicsError> {
    // Start from the sync shader so the resident path inherits every
    // struct definition, stub fn, rule body, dispatcher, and the
    // base 0..12 bindings verbatim. This keeps the two shaders in
    // lockstep — any new stub / const / struct added for sync flows
    // into resident automatically.
    //
    // Phase 3 Task 3.4: pass `has_gold_buf = true` so the gold-stub
    // section in `state_stub_fns` emits atomic bodies against the
    // binding-17 `gold_buf` declaration we append below. The sync
    // source passes `false` and keeps its no-op stubs.
    //
    // Task #79 SP-4: pass `has_standing_storage = true` so the
    // `state_adjust_standing` stub emits the real fold body against
    // the binding-18 / 19 standing storage. Sync source keeps the
    // no-op for modify_standing; cold_state_replay covers it.
    //
    // Subsystem 2 Phase 4 PR-4: pass `has_memory_storage = true`
    // so `state_push_agent_memory` writes into the binding-20 / 21
    // ring + cursor storage. Sync source keeps the no-op; CPU
    // cold_state_replay handles `RecordMemory` there.
    let mut out = build_physics_shader_with_chronicle(
        physics,
        ctx,
        event_ring_capacity,
        chronicle_ring_capacity,
        true,
        true,
        true,
        true,
    )?;

    // Append the resident-only bindings. These are additive — the sync
    // pipeline doesn't include them in its bind-group layout, but WGSL
    // allows unused bindings at the module level as long as no entry
    // point references them. We gate the references behind the
    // `cs_physics_resident` entry so the sync `cs_physics` entry keeps
    // its unchanged binding set.
    //
    // `indirect_args` is declared as `array<u32>` (NOT
    // `array<vec3<u32>>`) because WGSL pads `vec3<u32>` to 16-byte
    // stride in storage arrays, which would misalign the host's
    // 12-byte `IndirectArgs` layout and the 12-byte offset
    // `dispatch_workgroups_indirect` reads at. Indexing as
    // `indirect_args[slot*3 + N]` gives the tight 12-byte stride the
    // indirect-dispatch API expects.
    out.push_str(
        "\n// ---- Task B7 — resident-path bindings + entry point ----\n\
         @group(0) @binding(13) var<storage, read_write> indirect_args: array<u32>;\n\
         @group(0) @binding(14) var<storage, read_write> num_events_buf: array<u32>;\n\n\
         struct ResidentPhysicsCfg {\n\
         \x20 read_slot: u32,\n\
         \x20 write_slot: u32,\n\
         \x20 _pad0: u32,\n\
         \x20 _pad1: u32,\n\
         };\n\
         @group(0) @binding(15) var<uniform> resident_cfg: ResidentPhysicsCfg;\n\n",
    );

    // Resident entry point. Reads num_events from the indirect slot
    // chain instead of `cfg.num_events`, and at end-of-dispatch thread
    // 0 publishes the next iter's workgroup count + num_events into
    // slot `write_slot`. When `emitted == 0u`, we write
    // `(0u, 1u, 1u)` — subsequent `dispatch_workgroups_indirect` calls
    // become GPU no-ops, which is the convergence signal without a
    // readback.
    out.push_str(&format!(
        "@compute @workgroup_size({PHYSICS_WORKGROUP_SIZE})\n\
         fn cs_physics_resident(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \x20   let i = gid.x;\n\
         \x20   let num_events_this_iter = num_events_buf[resident_cfg.read_slot];\n\
         \x20   if (i < num_events_this_iter) {{\n\
         \x20       physics_dispatch(i);\n\
         \x20   }}\n\
         \x20   // End-of-kernel: thread 0 writes the next iter's wg +\n\
         \x20   // event count. Workgroup-scope barriers aren't required\n\
         \x20   // here because `atomicLoad` on `event_ring_tail` sees the\n\
         \x20   // post-dispatch tail; every emitter finishes its\n\
         \x20   // `atomicAdd` before the workgroup exits.\n\
         \x20   //\n\
         \x20   // NOTE: this assumes all emitter threads across all\n\
         \x20   // workgroups have also completed. A true global barrier\n\
         \x20   // is not available in WGSL, but in practice the driver\n\
         \x20   // serialises workgroups within a dispatch, and this\n\
         \x20   // kernel's single-writer thread is in the SAME dispatch.\n\
         \x20   // The indirect dispatch for the NEXT iter reads the new\n\
         \x20   // slot value AFTER this entire dispatch retires, via the\n\
         \x20   // wgpu command-buffer ordering guarantee. That ordering\n\
         \x20   // is what makes this correct without a cross-workgroup\n\
         \x20   // barrier.\n\
         \x20   if (i == 0u) {{\n\
         \x20       let emitted = atomicLoad(&event_ring_tail);\n\
         \x20       let wg_size = {PHYSICS_WORKGROUP_SIZE}u;\n\
         \x20       let cap_wg = (cfg.agent_cap + wg_size - 1u) / wg_size;\n\
         \x20       let requested = (emitted + wg_size - 1u) / wg_size;\n\
         \x20       var wg = requested;\n\
         \x20       if (wg > cap_wg) {{ wg = cap_wg; }}\n\
         \x20       // 12-byte stride layout: (x, y, z) at slot*3 + (0, 1, 2).\n\
         \x20       let base = resident_cfg.write_slot * 3u;\n\
         \x20       indirect_args[base + 0u] = wg;\n\
         \x20       indirect_args[base + 1u] = 1u;\n\
         \x20       indirect_args[base + 2u] = 1u;\n\
         \x20       num_events_buf[resident_cfg.write_slot] = emitted;\n\
         \x20   }}\n\
         }}\n",
    ));

    Ok(out)
}

/// Inject `let wgsl_world_tick = sim_cfg.tick;` right after the rule's
/// `fn name(event_idx: u32) {` header. The emitter produces that
/// identifier as a bare name in emit sites; declaring it as a function-
/// scope `let` keeps the emitter output unchanged.
///
/// Task 2.8 of the GPU sim-state refactor swapped the source from the
/// per-kernel `cfg.tick` uniform to the shared `SimCfg` storage buffer
/// so physics + scoring + movement + apply_actions all read the same
/// atomically-incremented tick. `sim_cfg.tick` is a plain (non-atomic)
/// read here — the seed-indirect kernel is the sole atomic writer, and
/// wgpu's compute-pass scheduling sequences the seed dispatch before
/// the physics dispatches that consume the post-increment value.
fn wrap_rule_with_tick_alias(wgsl: &str) -> String {
    // Find the first `{` after `fn physics_...(event_idx: u32)`.
    let mut out = String::with_capacity(wgsl.len() + 64);
    let mut injected = false;
    for line in wgsl.lines() {
        out.push_str(line);
        out.push('\n');
        if !injected && line.contains("fn physics_") && line.trim_end().ends_with('{') {
            out.push_str("    let wgsl_world_tick: u32 = sim_cfg.tick;\n");
            injected = true;
        }
    }
    out
}

/// Emit `EVENT_KIND_<SCREAMING_SNAKE>: u32 = N;` consts for every
/// event kind the emitter references. Mirrors `event_ring::EventKindTag`
/// ordinals.
fn event_kind_consts() -> String {
    // Mirrors `EventKindTag::raw()` values. Order here is informational —
    // the emitter references these by name, so only the (name, value)
    // pair matters.
    let entries: &[(&str, u32)] = &[
        ("AGENT_MOVED", 0),
        ("AGENT_ATTACKED", 1),
        ("AGENT_DIED", 2),
        ("AGENT_FLED", 3),
        ("AGENT_ATE", 4),
        ("AGENT_DRANK", 5),
        ("AGENT_RESTED", 6),
        ("AGENT_CAST", 7),
        ("AGENT_USED_ITEM", 8),
        ("AGENT_HARVESTED", 9),
        ("AGENT_PLACED_TILE", 10),
        ("AGENT_PLACED_VOXEL", 11),
        ("AGENT_HARVESTED_VOXEL", 12),
        ("AGENT_CONVERSED", 13),
        ("AGENT_SHARED_STORY", 14),
        ("AGENT_COMMUNICATED", 15),
        ("INFORMATION_REQUESTED", 16),
        ("AGENT_REMEMBERED", 17),
        ("QUEST_POSTED", 18),
        ("QUEST_ACCEPTED", 19),
        ("BID_PLACED", 20),
        ("ANNOUNCE_EMITTED", 21),
        ("RECORD_MEMORY", 22),
        ("OPPORTUNITY_ATTACK_TRIGGERED", 25),
        ("EFFECT_DAMAGE_APPLIED", 26),
        ("EFFECT_HEAL_APPLIED", 27),
        ("EFFECT_SHIELD_APPLIED", 28),
        ("EFFECT_STUN_APPLIED", 29),
        ("EFFECT_SLOW_APPLIED", 30),
        ("EFFECT_GOLD_TRANSFER", 31),
        ("EFFECT_STANDING_DELTA", 32),
        ("CAST_DEPTH_EXCEEDED", 33),
        ("ENGAGEMENT_COMMITTED", 34),
        ("ENGAGEMENT_BROKEN", 35),
        ("FEAR_SPREAD", 36),
        ("PACK_ASSIST", 37),
        ("RALLY_CALL", 38),
        // `CHRONICLE_ENTRY` is non-replayable (no tag in event_ring) —
        // the drain ignores the slot. Picking 24 (reserved by event_ring
        // as an unused slot) keeps the kind addressable in shader code
        // without colliding with any replayable event.
        ("CHRONICLE_ENTRY", 24),
    ];
    let mut out = String::new();
    for (name, val) in entries {
        out.push_str(&format!("const EVENT_KIND_{name}: u32 = {val}u;\n"));
    }
    out
}

/// Emit the state-stub fns the physics emitter references.
///
/// Phase 3 Task 3.4 — `has_gold_buf` selects whether the
/// `state_add_agent_gold` / `state_set_agent_gold` bodies are no-ops
/// (sync path — gold still runs via CPU `cold_state_replay`) or real
/// atomic mutations against the resident `gold_buf` side buffer
/// (resident batch path — CPU cold_inventory is rehydrated from GPU
/// on `snapshot()` after Task 3.5). The resident shader declares
/// binding 17 for `gold_buf` before including this prelude so the
/// real bodies compile.
///
/// Task #79 SP-4 — `has_standing_storage` selects whether the
/// `state_adjust_standing(a, b, delta)` body is a no-op (sync path,
/// standing still runs via CPU `cold_state_replay`) or a real
/// find-or-evict fold against bindings 18 / 19.
///
/// Subsystem 2 Phase 4 PR-4 — `has_memory_storage` selects whether
/// the `state_push_agent_memory(observer, source, payload, confidence,
/// tick)` body is a no-op (sync path — memory still runs via CPU
/// `cold_state_replay`) or a real monotonic-cursor ring push against
/// bindings 20 / 21. The stub quantises `confidence: f32` to q8 and
/// hardcodes `kind = 0` to match the CPU-generated rule at
/// `crates/engine/src/generated/physics/record_memory.rs`.
fn state_stub_fns(
    has_gold_buf: bool,
    has_standing_storage: bool,
    has_memory_storage: bool,
) -> String {
    // The emitter lists every stub name in the module doc. All scalar
    // getters/setters project into a single field of `agents[id - 1]`
    // (ids are 1-based; slot 0 is unused).
    //
    // `id` is bounds-checked — out-of-range ids return sentinel / are
    // no-ops so a malformed event can't corrupt state.
    let src = r#"
fn slot_of(id: u32) -> u32 {
    // AgentId is 1-based NonZeroU32. Dead slots / unspawned ids both
    // resolve to a slot whose `alive == 0`, so the caller's guard
    // against `alive(id)` handles them.
    if (id == 0u) { return 0xFFFFFFFFu; }
    let slot = id - 1u;
    if (slot >= cfg.agent_cap) { return 0xFFFFFFFFu; }
    return slot;
}

fn state_agent_alive(id: u32) -> bool {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return false; }
    return agents[s].alive != 0u;
}
fn state_agent_hp(id: u32) -> f32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0.0; }
    return agents[s].hp;
}
fn state_agent_max_hp(id: u32) -> f32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0.0; }
    return agents[s].max_hp;
}
fn state_agent_shield_hp(id: u32) -> f32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0.0; }
    return agents[s].shield_hp;
}
fn state_agent_attack_damage(id: u32) -> f32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0.0; }
    return agents[s].attack_damage;
}
fn state_agent_stun_expires_at(id: u32) -> u32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0u; }
    return agents[s].stun_expires_at;
}
fn state_agent_slow_expires_at(id: u32) -> u32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0u; }
    return agents[s].slow_expires_at;
}
fn state_agent_slow_factor_q8(id: u32) -> i32 {
    // Slow factor is stored as i16 on CPU; packed as the low 16 bits of
    // a u32 on upload. Sign-extend the 16-bit value into i32 so the
    // compiled physics bodies (which pair this with an i32-typed
    // `factor_q8` payload binding) get consistent signed arithmetic.
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0; }
    let raw = agents[s].slow_factor_q8 & 0xFFFFu;
    // Sign-extend: if top bit of the 16-bit field is set, extend ones.
    if ((raw & 0x8000u) != 0u) {
        return bitcast<i32>(raw | 0xFFFF0000u);
    }
    return bitcast<i32>(raw);
}
// Gold is not part of the GPU agent SoA — stub returns 0 and the
// corresponding `state_add_agent_gold` / `_set_agent_gold` below are
// no-ops. Callers that need gold continue to run the CPU cascade.
fn state_agent_gold(id: u32) -> i32 { return 0; }
fn state_agent_cooldown_next_ready(id: u32) -> u32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return 0u; }
    return agents[s].cooldown_next_ready;
}
fn state_agent_engaged_with(id: u32) -> u32 {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return ENGAGED_SENTINEL; }
    return agents[s].engaged_with;
}

fn state_set_agent_hp(id: u32, v: f32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].hp = v;
}
fn state_set_agent_shield_hp(id: u32, v: f32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].shield_hp = v;
}
fn state_set_agent_stun_expires_at(id: u32, v: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].stun_expires_at = v;
}
fn state_set_agent_slow_expires_at(id: u32, v: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].slow_expires_at = v;
}
fn state_set_agent_slow_factor_q8(id: u32, v: i32) {
    // Re-pack signed i32 into the u32 slot as low 16 bits. CPU readback
    // strips the high bits and reinterprets as i16.
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].slow_factor_q8 = bitcast<u32>(v) & 0xFFFFu;
}
// ---- GOLD_STUBS_PLACEHOLDER ----
// ---- STANDING_STUB_PLACEHOLDER ----
fn state_set_agent_cooldown_next_ready(id: u32, v: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].cooldown_next_ready = v;
}
fn state_set_agent_engaged_with(id: u32, partner: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].engaged_with = partner;
}
fn state_clear_agent_engaged_with(id: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].engaged_with = ENGAGED_SENTINEL;
}
fn state_kill_agent(id: u32) {
    let s = slot_of(id);
    if (s == 0xFFFFFFFFu) { return; }
    agents[s].alive = 0u;
    // Intentionally NOT clearing `engaged_with` here — the CPU's
    // `SimState::kill_agent` doesn't touch the engagement field either.
    // The `engagement_on_death` physics rule (triggered by the
    // AgentDied event) is what tears down the pairing + emits
    // EngagementBroken. If `state_kill_agent` wipes engaged_with
    // pre-cascade, `engagement_on_death` sees a self-engaged sentinel
    // and skips the teardown — a silent miss of the EngagementBroken
    // event the CPU cascade emits.
}
// ---- MEMORY_STUB_PLACEHOLDER ----
"#;

    // Substitute the gold-stub placeholder with either no-op bodies
    // (sync path — gold still runs via CPU `cold_state_replay`) or
    // real atomic bodies (resident path — Phase 3 Task 3.4).
    let gold_bodies = if has_gold_buf {
        "// Phase 3 Task 3.4 — real gold mutations against the resident\n\
         // `gold_buf` side buffer (binding 17, one atomic<i32> per slot).\n\
         // Sync path keeps the no-op bodies below; only the resident\n\
         // shader builder (`build_physics_shader_resident`) passes\n\
         // `has_gold_buf = true` so these atomic calls compile.\n\
         fn state_add_agent_gold(id: u32, delta: i32) {\n\
         \x20   let s = slot_of(id);\n\
         \x20   if (s == 0xFFFFFFFFu) { return; }\n\
         \x20   atomicAdd(&gold_buf[s], delta);\n\
         }\n\
         fn state_set_agent_gold(id: u32, v: i32) {\n\
         \x20   let s = slot_of(id);\n\
         \x20   if (s == 0xFFFFFFFFu) { return; }\n\
         \x20   atomicStore(&gold_buf[s], v);\n\
         }\n"
    } else {
        "// Gold mutations: no-op on GPU for the sync path. The\n\
         // transfer_gold / ModifyStanding rules still fire on the CPU\n\
         // side (via `cold_state_replay`); the GPU path preserves state\n\
         // by leaving the (absent) gold buffer untouched. The resident\n\
         // batch path emits the real atomic bodies against binding 17.\n\
         fn state_set_agent_gold(id: u32, v: i32) { }\n\
         fn state_add_agent_gold(id: u32, delta: i32) { }\n"
    };
    // Task #79 SP-4 — substitute the standing-stub placeholder.
    // Real body implements find-or-reserve-or-evict-weakest against
    // the standing_records_buf + standing_counts_buf (bindings 18 /
    // 19). Matches the Phase-1 emitter's canonical WGSL fold body at
    // `crates/dsl_compiler/src/emit_view_wgsl.rs:1123-1475` — inlined
    // here because the DSL emitter produces a standalone fold kernel
    // entry point; physics needs the logic callable as a function
    // from within `state_adjust_standing(a, b, delta)`.
    let standing_body = if has_standing_storage {
        "// Task #79 SP-4 — real standing fold against bindings 18/19.\n\
         // Canonicalises (a, b) → (owner=min, other=max), scans the\n\
         // owner's row for an existing slot (update-in-place with clamp),\n\
         // reserves an empty slot via atomicAdd on counts, else evicts\n\
         // the smallest-|value| slot if |delta| beats it. Mirrors the\n\
         // CPU Standing::adjust semantics byte-for-byte.\n\
         fn state_adjust_standing(a: u32, b: u32, delta: i32) {\n\
         \x20   if (a == 0u || b == 0u) { return; }\n\
         \x20   let owner = min(a, b);\n\
         \x20   let other_id = max(a, b);\n\
         \x20   let owner_slot = owner - 1u;\n\
         \x20   if (owner_slot >= cfg.agent_cap) { return; }\n\
         \x20   let row_base = owner_slot * STANDING_K;\n\
         \x20\n\
         \x20   // 1. Find existing: scan the occupied prefix for\n\
         \x20   //    matching `other`. Update in-place + clamp + stamp.\n\
         \x20   let count = atomicLoad(&standing_counts_buf[owner_slot]);\n\
         \x20   let scan_len = min(count, STANDING_K);\n\
         \x20   var found_idx: u32 = 0xFFFFFFFFu;\n\
         \x20   for (var i: u32 = 0u; i < scan_len; i = i + 1u) {\n\
         \x20       if (standing_records_buf[row_base + i].other == other_id) {\n\
         \x20           found_idx = i;\n\
         \x20           break;\n\
         \x20       }\n\
         \x20   }\n\
         \x20   if (found_idx != 0xFFFFFFFFu) {\n\
         \x20       let cur = standing_records_buf[row_base + found_idx].value;\n\
         \x20       var updated = cur + delta;\n\
         \x20       if (updated > STANDING_CLAMP_POS) { updated = STANDING_CLAMP_POS; }\n\
         \x20       if (updated < STANDING_CLAMP_NEG) { updated = STANDING_CLAMP_NEG; }\n\
         \x20       standing_records_buf[row_base + found_idx].value = updated;\n\
         \x20       standing_records_buf[row_base + found_idx].anchor_tick = sim_cfg.tick;\n\
         \x20       return;\n\
         \x20   }\n\
         \x20\n\
         \x20   // 2. Reserve-empty: atomicAdd on counts. Losers (index\n\
         \x20   //    >= K) fall through to evict.\n\
         \x20   var clamped_delta: i32 = delta;\n\
         \x20   if (clamped_delta > STANDING_CLAMP_POS) { clamped_delta = STANDING_CLAMP_POS; }\n\
         \x20   if (clamped_delta < STANDING_CLAMP_NEG) { clamped_delta = STANDING_CLAMP_NEG; }\n\
         \x20   if (count < STANDING_K) {\n\
         \x20       let new_idx = atomicAdd(&standing_counts_buf[owner_slot], 1u);\n\
         \x20       if (new_idx < STANDING_K) {\n\
         \x20           standing_records_buf[row_base + new_idx].other       = other_id;\n\
         \x20           standing_records_buf[row_base + new_idx].value       = clamped_delta;\n\
         \x20           standing_records_buf[row_base + new_idx].anchor_tick = sim_cfg.tick;\n\
         \x20           standing_records_buf[row_base + new_idx]._pad        = 0u;\n\
         \x20           return;\n\
         \x20       }\n\
         \x20       // Lost the reserve race — counts overshot by one; benign\n\
         \x20       // because future scans use `min(count, K)`. Fall\n\
         \x20       // through to evict.\n\
         \x20   }\n\
         \x20\n\
         \x20   // 3. Evict smallest-|value| slot if |delta| beats it.\n\
         \x20   var weakest_idx: u32 = 0u;\n\
         \x20   var weakest_mag: i32 = abs(standing_records_buf[row_base].value);\n\
         \x20   for (var i: u32 = 1u; i < STANDING_K; i = i + 1u) {\n\
         \x20       let mag = abs(standing_records_buf[row_base + i].value);\n\
         \x20       if (mag < weakest_mag) {\n\
         \x20           weakest_mag = mag;\n\
         \x20           weakest_idx = i;\n\
         \x20       }\n\
         \x20   }\n\
         \x20   if (abs(clamped_delta) > weakest_mag) {\n\
         \x20       standing_records_buf[row_base + weakest_idx].other       = other_id;\n\
         \x20       standing_records_buf[row_base + weakest_idx].value       = clamped_delta;\n\
         \x20       standing_records_buf[row_base + weakest_idx].anchor_tick = sim_cfg.tick;\n\
         \x20       standing_records_buf[row_base + weakest_idx]._pad        = 0u;\n\
         \x20   }\n\
         }\n"
    } else {
        "// Standing mutations: no-op on GPU for the sync path.\n\
         // modify_standing still runs via CPU cold_state_replay. The\n\
         // resident batch path emits the real fold body against\n\
         // bindings 18 / 19 (Task #79 SP-4).\n\
         fn state_adjust_standing(a: u32, b: u32, delta: i32) { }\n"
    };

    // Subsystem 2 Phase 4 PR-4 — substitute the memory-stub
    // placeholder. Real body quantises `confidence: f32` to q8,
    // hardcodes `kind = 0` (matching CPU `record_memory`), splits
    // the event's `payload: u32` into `payload_lo = payload,
    // payload_hi = 0` (DSL lowering today passes only the low word
    // of the `u64 fact_payload` — documented truncation), and writes
    // the record at `owner_slot * MEMORY_K + (cursor % MEMORY_K)`
    // with an `atomicAdd` bumping the cursor. Monotonic cursor +
    // modulo makes this naturally race-safe (no CAS loop, no evict
    // scan).
    let memory_body = if has_memory_storage {
        "// Subsystem 2 Phase 4 PR-4 — real memory ring push against\n\
         // bindings 20/21. Monotonic cursor + slot = cursor % K gives\n\
         // FIFO semantics with natural eviction at K overflow.\n\
         fn state_push_agent_memory(observer: u32, source: u32, payload: u32, confidence: f32, t: u32) {\n\
         \x20   let s = slot_of(observer);\n\
         \x20   if (s == 0xFFFFFFFFu) { return; }\n\
         \x20\n\
         \x20   // Reserve a slot by atomically bumping the owner's cursor.\n\
         \x20   // The returned value is this write's absolute index; slot\n\
         \x20   // = idx % MEMORY_K. Concurrent writes on the same owner\n\
         \x20   // serialise through this atomicAdd and each take a\n\
         \x20   // distinct slot modulo K.\n\
         \x20   let idx = atomicAdd(&memory_cursors_buf[s], 1u);\n\
         \x20   let ring_slot = s * MEMORY_K + (idx % MEMORY_K);\n\
         \x20\n\
         \x20   // Quantise confidence to q8. CPU `record_memory` does\n\
         \x20   // `(c.clamp(0.0, 1.0) * 255.0) as u8` — mirror it here.\n\
         \x20   var c = confidence;\n\
         \x20   if (c < 0.0) { c = 0.0; }\n\
         \x20   if (c > 1.0) { c = 1.0; }\n\
         \x20   let conf_q8: u32 = u32(c * 255.0);\n\
         \x20\n\
         \x20   memory_records_buf[ring_slot].source     = source;\n\
         \x20   memory_records_buf[ring_slot].kind       = 0u;\n\
         \x20   memory_records_buf[ring_slot].payload_lo = payload;\n\
         \x20   memory_records_buf[ring_slot].payload_hi = 0u;\n\
         \x20   memory_records_buf[ring_slot].confidence = conf_q8;\n\
         \x20   memory_records_buf[ring_slot].tick       = t;\n\
         }\n"
    } else {
        "// Memory push: no-op on GPU for the sync path. The\n\
         // record_memory rule still fires on the CPU side (via\n\
         // cold_state_replay); the GPU path preserves state by\n\
         // leaving the (absent) memory ring storage untouched. The\n\
         // resident batch path emits the real ring-push body against\n\
         // bindings 20 / 21.\n\
         fn state_push_agent_memory(observer: u32, source: u32, payload: u32, confidence: f32, t: u32) { }\n"
    };

    src.replace("// ---- GOLD_STUBS_PLACEHOLDER ----", gold_bodies)
        .replace("// ---- STANDING_STUB_PLACEHOLDER ----", standing_body)
        .replace("// ---- MEMORY_STUB_PLACEHOLDER ----", memory_body)
}

/// Emit spatial-query wrapper fns. The emitter calls
/// `spatial_nearest_hostile_to(agent, radius)` / `_nearby_kin_count` /
/// `_nearby_kin_at` — we back them with pre-computed per-slot result
/// buffers. `radius` is ignored; the driver pre-computes with the
/// relevant radius based on which rule it expects to fire (12 m kin,
/// engagement_range for hostile).
fn spatial_stub_fns() -> String {
    r#"
fn spatial_nearest_hostile_to(agent: u32, radius: f32) -> u32 {
    let s = slot_of(agent);
    if (s == 0xFFFFFFFFu) { return 0xFFFFFFFFu; }
    return nearest_hostile_buf[s];
}
fn spatial_nearby_kin_count(agent: u32, radius: f32) -> u32 {
    let s = slot_of(agent);
    if (s == 0xFFFFFFFFu) { return 0u; }
    return kin_lists[s].count;
}
fn spatial_nearby_kin_at(agent: u32, radius: f32, idx: u32) -> u32 {
    let s = slot_of(agent);
    if (s == 0xFFFFFFFFu) { return 0xFFFFFFFFu; }
    if (idx >= PHYSICS_SPATIAL_K) { return 0xFFFFFFFFu; }
    return kin_lists[s].ids[idx];
}
"#.to_string()
}

// ---------------------------------------------------------------------------
// Physics kernel
// ---------------------------------------------------------------------------

/// Compiled physics kernel + per-run buffer pool.
///
/// One pipeline + one bind group layout. Buffers are sized by agent_cap
/// at first run and reused across runs with the same cap.
pub struct PhysicsKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Task B7 — resident pipeline + BGL. Separate from the sync
    /// pipeline so the sync path's bind-group shape stays unchanged.
    /// The resident BGL extends the sync BGL with bindings 13/14/15:
    /// `indirect_args` (storage rw), `num_events_buf` (storage rw),
    /// `resident_cfg` (uniform).
    pipeline_resident: wgpu::ComputePipeline,
    bind_group_layout_resident: wgpu::BindGroupLayout,
    /// Event ring owned by the kernel — physics writes into this ring
    /// and the driver drains it after each dispatch.
    event_ring: GpuEventRing,
    /// Task 203 — chronicle ring owned by the kernel. Dedicated buffer
    /// for `emit ChronicleEntry` records so the main event ring stays
    /// free of observability traffic. The cascade driver does NOT
    /// drain this ring per tick; callers opt in via
    /// [`GpuBackend::flush_chronicle`].
    chronicle_ring: GpuChronicleRing,
    pool: Option<BufferPool>,
    /// Task B7 — resident-path pool. Holds cfg uniforms (main +
    /// resident), scratch spatial / ability buffers that the resident
    /// path still needs to bind even though caller supplies agents /
    /// events. Separate from `pool` so the sync path's allocations
    /// aren't perturbed by resident-only callers.
    pool_resident: Option<ResidentBufferPool>,
    /// Resident-path bind-group cache. Keyed by the 13 stable buffer
    /// identities plus the 3 iter-dependent ones (events_in +
    /// event_ring.records + event_ring.tail). Each cascade tick hits
    /// at most 3 distinct keys (iter 0, odd iters ≥ 1, even iters ≥ 2)
    /// and the ones in slot are stable across ticks in a batch. The
    /// HashMap thus has ≤ 3 entries and hits 100% after iter 0 of
    /// tick 1. Keyed by hashing the wgpu Buffer handles (`Buffer: Eq +
    /// Hash` via its internal Arc pointer — see wgpu-26's
    /// `impl_eq_ord_hash_proxy!`).
    resident_bg_cache: std::collections::HashMap<ResidentBgKey, wgpu::BindGroup>,
    /// Perf Stage A.2 — running hit/miss counter for
    /// `resident_bg_cache`. Incremented on every lookup in
    /// `run_batch_resident`. A cold start hits ~3 misses per tick
    /// (one per unique (events_in, events_out, resident_cfg) triple:
    /// iter 0, odd iters ≥ 1, even iters ≥ 2) for the first tick, then
    /// 100% hits thereafter. `resident_bg_cache_stats()` surfaces the
    /// numbers so perf tests can flag a thrash bug (e.g. 400 misses
    /// across 400 lookups = keyed-identity drift).
    resident_bg_cache_hits:   u64,
    resident_bg_cache_misses: u64,
    /// Capacity the ring was provisioned with. Retained for diagnostics
    /// (Piece 3 may surface it in run reports); currently unread so the
    /// `allow(dead_code)` keeps the compiler happy.
    #[allow(dead_code)]
    event_ring_capacity: u32,
    /// Chronicle ring capacity — retained for pipeline-time validation
    /// of the shader constant `CHRONICLE_RING_CAP` matches the host
    /// buffer size.
    #[allow(dead_code)]
    chronicle_ring_capacity: u32,
}

/// Task B7 — resident-path pool. Holds ONLY what the resident
/// dispatch cannot get from caller-supplied buffers: the main
/// `PhysicsCfg` uniform and the new `ResidentPhysicsCfg` uniform
/// (read_slot / write_slot).
///
/// Agents / abilities / kin / nearest-hostile / events / event ring
/// all flow in as caller-supplied buffers, matching the B5/B6 pattern
/// where the resident caller owns the big state arrays.
struct ResidentBufferPool {
    agent_cap: u32,
    cfg_buf: wgpu::Buffer,
    /// Per-iteration uniform buffers for `ResidentPhysicsCfg`.
    /// One buffer per iteration slot (indexed by `read_slot`), with
    /// its contents written ONCE at pool-creation time and stable
    /// across every tick in the batch.
    ///
    /// **Why one buffer per iteration (not one buffer written N times
    /// with per-iter offsets or `queue.write_buffer`):** the prior
    /// implementation wrote a single `resident_cfg_buf` via
    /// `queue.write_buffer(..., 0, ...)` once per cascade iteration,
    /// each call with a different `{read_slot, write_slot}` pair.
    /// `queue.write_buffer` writes to the same byte range COLLAPSE —
    /// only the last write lands when the submit actually begins, so
    /// every physics iteration ended up reading the FINAL iteration's
    /// uniform. That meant iter 0 saw `read_slot=N-1` and read
    /// `num_events_buf[N-1]` (which is 0), skipping the AgentAttacked
    /// records at slot 0 and leaving the chronicle ring untouched.
    /// See `fix(engine_gpu): step_batch chronicle emit — resident_cfg
    /// uniform queue.write_buffer collapse` for the full diagnosis.
    ///
    /// With one buffer per iteration, each iter's bind group
    /// references a distinct buffer identity — no queue collapse,
    /// no per-iter uploads either (contents are static because
    /// `read_slot = iter` and `write_slot = iter + 1`).
    resident_cfg_bufs: Vec<wgpu::Buffer>,
    /// Last cfg uploaded via `queue.write_buffer(&cfg_buf, ...)`. The
    /// driver calls `run_batch_resident` 8× per tick (cascade
    /// iterations) with the same cfg each time; deduping saves 7 of
    /// those 8 writes/tick. `None` = never written. Invalidated on
    /// pool rebuild (new `cfg_buf`).
    last_cfg: Option<PhysicsCfg>,
}

/// Cache key for `PhysicsKernel::resident_bg_cache`. Combines
/// agent_cap with the wgpu buffer identities of every caller-supplied
/// resident binding. `wgpu::Buffer: Eq + Hash` is derived from its
/// internal Arc pointer (see wgpu-26's `impl_eq_ord_hash_proxy!`), so
/// the key compares "same underlying buffer handle" rather than byte
/// contents — exactly what the cache wants.
#[derive(Clone, Eq, Hash, PartialEq)]
struct ResidentBgKey {
    agent_cap: u32,
    agents: wgpu::Buffer,
    ability_known: wgpu::Buffer,
    ability_cooldown: wgpu::Buffer,
    ability_effects_count: wgpu::Buffer,
    ability_effects: wgpu::Buffer,
    kin: wgpu::Buffer,
    nearest_hostile: wgpu::Buffer,
    events_in: wgpu::Buffer,
    event_ring_records: wgpu::Buffer,
    event_ring_tail: wgpu::Buffer,
    chronicle_records: wgpu::Buffer,
    chronicle_tail: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    num_events: wgpu::Buffer,
    /// Per-iteration resident_cfg uniform handle — keys the BG cache
    /// by iteration so iter N doesn't alias iter (N-1)'s cached BG.
    resident_cfg: wgpu::Buffer,
    /// Task 2.8 — caller-supplied shared `SimCfg` buffer. The backend
    /// holds one resident handle per `GpuBackend`, so this key is
    /// stable across every cascade iteration in a batch (the cache
    /// still hits 100% after iter 0).
    sim_cfg: wgpu::Buffer,
    /// Phase 3 Task 3.4 — resident gold ledger side buffer. Like
    /// `sim_cfg`, stable across every cascade iteration in a batch
    /// (single handle on the backend, re-allocated only when
    /// `agent_cap` grows), so this key is cache-friendly.
    gold_buf: wgpu::Buffer,
    /// Task #79 SP-4 — resident standing view storage (records +
    /// counts). Same stability story as `gold_buf` — one pair of
    /// handles per `GpuBackend`, stable across every cascade
    /// iteration in a batch.
    standing_records: wgpu::Buffer,
    standing_counts:  wgpu::Buffer,
    /// Subsystem 2 Phase 4 PR-4 — resident memory view storage
    /// (records + cursors). Same stability story as the standing
    /// buffers above — one pair of handles per `GpuBackend`, stable
    /// across every cascade iteration in a batch.
    memory_records: wgpu::Buffer,
    memory_cursors: wgpu::Buffer,
    /// Per-tick alive bitmap — single handle per `GpuBackend`, stable
    /// across every cascade iteration in a batch.
    alive_bitmap: wgpu::Buffer,
}

struct BufferPool {
    agent_cap: u32,
    agents_buf: wgpu::Buffer,
    agents_readback: wgpu::Buffer,
    abilities_known_buf: wgpu::Buffer,
    abilities_cooldown_buf: wgpu::Buffer,
    abilities_effects_count_buf: wgpu::Buffer,
    abilities_effects_buf: wgpu::Buffer,
    kin_lists_buf: wgpu::Buffer,
    nearest_hostile_buf: wgpu::Buffer,
    events_in_buf: wgpu::Buffer,
    events_in_capacity: u32,
    cfg_buf: wgpu::Buffer,
    /// Task 2.8 — pool-owned `SimCfg` buffer used only by the sync
    /// `run_batch` path so it stays self-contained without a caller-
    /// supplied resident SimCfg. Refreshed each `run_batch` call via
    /// `SimCfg::from_state(state)` + `upload_sim_cfg`. The resident
    /// path binds the caller's buffer instead and does not read this.
    sync_sim_cfg_buf: wgpu::Buffer,
    /// Alive bitmap for the sync path. Packed host-side from
    /// `agent_slots_in` on each `run_batch` call and uploaded. The
    /// resident path uses its own caller-supplied bitmap populated
    /// by `alive_pack_kernel`; this field is used only by sync.
    sync_alive_bitmap_buf: wgpu::Buffer,
    /// Persistent staging for the event-ring tail readback. Phase 9
    /// (task 195): sized once at init to avoid per-tick buffer alloc.
    drain_tail_staging: wgpu::Buffer,
    /// Persistent staging for the event-ring records readback. Phase 9
    /// (task 195): sized to the ring's capacity at init so each drain
    /// reuses the same staging rather than allocating 2.6 MB of
    /// throwaway readback per cascade iteration.
    drain_records_staging: wgpu::Buffer,
    /// Ring capacity the drain staging was sized for. Validated by
    /// `drain_raw_records_pooled` so a ring resize doesn't silently
    /// read partial records.
    drain_ring_capacity: u32,
}

impl PhysicsKernel {
    /// Build the physics kernel. Parses every rule from
    /// `assets/sim/physics.sim` (via the in-memory `PhysicsIR` list the
    /// caller supplies), emits WGSL, compiles, caches the pipeline.
    pub fn new(
        device: &wgpu::Device,
        physics: &[PhysicsIR],
        ctx: &EmitContext<'_>,
        event_ring_capacity: u32,
    ) -> Result<Self, PhysicsError> {
        Self::new_with_chronicle(
            device,
            physics,
            ctx,
            event_ring_capacity,
            DEFAULT_CHRONICLE_CAPACITY,
        )
    }

    /// Task 203 — explicit-capacity variant that lets callers size the
    /// chronicle ring independently of the main event ring. Default
    /// callers use [`PhysicsKernel::new`] which picks
    /// [`DEFAULT_CHRONICLE_CAPACITY`] (1 M records).
    pub fn new_with_chronicle(
        device: &wgpu::Device,
        physics: &[PhysicsIR],
        ctx: &EmitContext<'_>,
        event_ring_capacity: u32,
        chronicle_ring_capacity: u32,
    ) -> Result<Self, PhysicsError> {
        let wgsl = build_physics_shader_with_chronicle(
            physics,
            ctx,
            event_ring_capacity,
            chronicle_ring_capacity,
            false, // sync pipeline: no gold_buf binding
            false, // sync pipeline: no standing_storage binding
            false, // sync pipeline: no memory_storage binding
            true,  // sync + resident: alive_bitmap at slot 22 (host-packed on sync path)
        )?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::physics::wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(PhysicsError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{wgsl}"
            )));
        }

        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl_entries = [
            storage_rw(0), // agents
            storage_ro(1), // abilities_known
            storage_ro(2), // abilities_cooldown
            storage_ro(3), // abilities_effects_count
            storage_ro(4), // abilities_effects
            storage_ro(5), // kin_lists
            storage_ro(6), // nearest_hostile
            storage_ro(7), // events_in
            storage_rw(8), // event_ring records
            storage_rw(9), // event_ring tail
            uniform(10),   // cfg
            // Task 203 — dedicated chronicle ring. Physics routes
            // `emit ChronicleEntry` here instead of the main event ring.
            storage_rw(11), // chronicle_ring records
            storage_rw(12), // chronicle_ring tail
            // Task 2.8 — shared SimCfg storage buffer (world-scalars:
            // tick, engagement_range, cascade_max_iterations, ...).
            // Read-only; the seed-indirect kernel is the sole writer
            // and runs in a prior dispatch.
            storage_ro(SIM_CFG_BINDING),
            // Per-tick alive bitmap. Sync path: host-packed on every
            // `run_batch` call from `agent_slots_in`. Resident path:
            // written by `alive_pack_kernel` before the cascade runs.
            storage_ro(crate::alive_bitmap::ALIVE_BITMAP_BINDING),
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::physics::bgl"),
            entries: &bgl_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::physics::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::physics::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_physics"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- Task B7 — resident pipeline + BGL ----
        //
        // Built from a DIFFERENT shader source (the sync source plus
        // the 13/14/15 bindings + cs_physics_resident entry point).
        // A separate shader module is mandatory because the sync
        // shader source has no declaration for bindings 13-15 — adding
        // them here would perturb the sync shader bytes. Keeping the
        // resident path in its own module isolates the WGSL epilogue
        // from the sync byte-identity contract.
        let resident_wgsl = build_physics_shader_resident(
            physics,
            ctx,
            event_ring_capacity,
            chronicle_ring_capacity,
        )?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader_resident = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::physics::wgsl_resident"),
            source: wgpu::ShaderSource::Wgsl(resident_wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(PhysicsError::ShaderCompile(format!(
                "resident: {err}\n--- WGSL source ---\n{resident_wgsl}"
            )));
        }
        let bgl_entries_resident = [
            storage_rw(0),  // agents
            storage_ro(1),  // abilities_known
            storage_ro(2),  // abilities_cooldown
            storage_ro(3),  // abilities_effects_count
            storage_ro(4),  // abilities_effects
            storage_ro(5),  // kin_lists
            storage_ro(6),  // nearest_hostile
            storage_ro(7),  // events_in
            storage_rw(8),  // event_ring records
            storage_rw(9),  // event_ring tail
            uniform(10),    // cfg
            storage_rw(11), // chronicle_ring records
            storage_rw(12), // chronicle_ring tail
            storage_rw(13), // indirect_args
            storage_rw(14), // num_events_buf
            uniform(15),    // resident_cfg
            // Task 2.8 — shared SimCfg storage buffer. Same binding
            // index as the sync BGL so both WGSL entry points reference
            // the identical `@binding(SIM_CFG_BINDING)` declaration.
            storage_ro(SIM_CFG_BINDING),
            // Phase 3 Task 3.4 — gold ledger side buffer, one
            // atomic<i32> per agent slot. transfer_gold DSL rules mutate
            // this via `state_add_agent_gold` / `state_set_agent_gold`
            // on the resident path.
            storage_rw(17),
            // Task #79 SP-4 — standing view storage.
            // 18: records `array<StandingEdgeGpu>` (storage, rw).
            // 19: counts  `array<atomic<u32>>` (storage, rw).
            // modify_standing DSL rule mutates via the real
            // `state_adjust_standing` body (find-or-evict fold).
            storage_rw(18),
            storage_rw(19),
            // Subsystem 2 Phase 4 PR-4 — memory view storage.
            // 20: records `array<MemoryEventGpu>` (storage, rw).
            // 21: cursors `array<atomic<u32>>` (storage, rw).
            // record_memory DSL rule mutates via the real
            // `state_push_agent_memory` body (monotonic ring push).
            storage_rw(20),
            storage_rw(21),
            // Per-tick alive bitmap at slot 22. Packed by
            // `alive_pack_kernel` at the top of every `step_batch`
            // tick; read by every `agents.alive(x)` lowering site.
            storage_ro(crate::alive_bitmap::ALIVE_BITMAP_BINDING),
        ];
        let bind_group_layout_resident =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine_gpu::physics::bgl_resident"),
                entries: &bgl_entries_resident,
            });
        let pipeline_layout_resident = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::physics::pl_resident"),
            bind_group_layouts: &[&bind_group_layout_resident],
            push_constant_ranges: &[],
        });
        let pipeline_resident =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("engine_gpu::physics::cp_resident"),
                layout: Some(&pipeline_layout_resident),
                module: &shader_resident,
                entry_point: Some("cs_physics_resident"),
                compilation_options: Default::default(),
                cache: None,
            });

        let event_ring = GpuEventRing::new(device, event_ring_capacity);
        let chronicle_ring = GpuChronicleRing::new(device, chronicle_ring_capacity);

        Ok(Self {
            pipeline,
            bind_group_layout,
            pipeline_resident,
            bind_group_layout_resident,
            event_ring,
            chronicle_ring,
            pool: None,
            pool_resident: None,
            resident_bg_cache: std::collections::HashMap::new(),
            resident_bg_cache_hits:   0,
            resident_bg_cache_misses: 0,
            event_ring_capacity,
            chronicle_ring_capacity,
        })
    }

    /// Perf Stage A.2 — running (hits, misses) counts for the
    /// resident bind-group cache. Post-warmup the cache should hit
    /// 100% after the first tick's ≤3 unique (events_in, events_out,
    /// resident_cfg) triples populate it; regressions show up as
    /// elevated miss counts relative to the total lookup count
    /// (`hits + misses`).
    ///
    /// Hidden from the top-level docs (consumers are perf tests + the
    /// research doc's refresh); access via
    /// `GpuBackend::physics_resident_bg_cache_stats()`.
    #[doc(hidden)]
    pub fn resident_bg_cache_stats(&self) -> (u64, u64) {
        (self.resident_bg_cache_hits, self.resident_bg_cache_misses)
    }

    /// Borrow the chronicle ring. `GpuBackend::flush_chronicle` drains
    /// this ring on demand; the cascade driver leaves it untouched so
    /// it accumulates across ticks until the caller explicitly drains.
    pub fn chronicle_ring(&self) -> &GpuChronicleRing {
        &self.chronicle_ring
    }

    pub fn chronicle_ring_mut(&mut self) -> &mut GpuChronicleRing {
        &mut self.chronicle_ring
    }

    /// Borrow the event ring — the cascade driver (Piece 3) reads the
    /// post-dispatch events out of this ring and feeds them back in as
    /// `events_in` for the next iteration.
    pub fn event_ring(&self) -> &GpuEventRing {
        &self.event_ring
    }

    pub fn event_ring_mut(&mut self) -> &mut GpuEventRing {
        &mut self.event_ring
    }

    fn ensure_pool(
        &mut self,
        device: &wgpu::Device,
        agent_cap: u32,
        events_in_capacity: u32,
    ) {
        let want_agent_cap = agent_cap.max(1);
        let want_events_cap = events_in_capacity.max(1);
        if let Some(p) = &self.pool {
            if p.agent_cap == want_agent_cap && p.events_in_capacity >= want_events_cap {
                return;
            }
        }

        let slot_bytes = (want_agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64);
        let agents_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::agents"),
            size: slot_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let agents_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::agents_rb"),
            size: slot_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let abilities_known_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::abilities_known"),
            size: (MAX_ABILITIES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let abilities_cooldown_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::abilities_cooldown"),
            size: (MAX_ABILITIES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let abilities_effects_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::abilities_effects_count"),
            size: (MAX_ABILITIES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let effect_bytes =
            (MAX_ABILITIES * MAX_EFFECTS) as u64 * (std::mem::size_of::<GpuEffectOp>() as u64);
        let abilities_effects_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::abilities_effects"),
            size: effect_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kin_bytes = (want_agent_cap as u64) * (std::mem::size_of::<GpuKinList>() as u64);
        let kin_lists_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::kin_lists"),
            size: kin_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nearest_hostile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::nearest_hostile"),
            size: (want_agent_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let event_bytes =
            (want_events_cap as u64) * (std::mem::size_of::<EventRecord>() as u64);
        let events_in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::events_in"),
            size: event_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::cfg"),
            size: std::mem::size_of::<PhysicsCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Task 2.8 — sync-path fallback SimCfg buffer. Owned by the
        // pool so `run_batch` stays self-contained; `SimCfg::from_state`
        // is called per dispatch and the result is uploaded here before
        // the bind group binds it at `SIM_CFG_BINDING`.
        let sync_sim_cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::sync_sim_cfg"),
            size: std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Alive bitmap for the sync path — one bit per agent slot,
        // packed host-side on every `run_batch` call from the
        // `agent_slots_in` slice + uploaded.
        let sync_alive_bitmap_buf = crate::alive_bitmap::create_alive_bitmap_buffer(
            device,
            want_agent_cap,
        );

        // Phase 9 (task 195): persistent drain staging. Previously
        // allocated per-drain, which at MAX_CASCADE_ITERATIONS=8 iterations
        // meant 8 fresh 2.6 MB buffers + 8 × 4 B tail buffers per tick.
        let ring_cap = self.event_ring_capacity.max(1);
        let drain_records_bytes =
            (ring_cap as u64) * (std::mem::size_of::<EventRecord>() as u64);
        let drain_tail_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::drain_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let drain_records_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::drain_records_staging"),
            size: drain_records_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.pool = Some(BufferPool {
            agent_cap: want_agent_cap,
            agents_buf,
            agents_readback,
            abilities_known_buf,
            abilities_cooldown_buf,
            abilities_effects_count_buf,
            abilities_effects_buf,
            kin_lists_buf,
            nearest_hostile_buf,
            events_in_buf,
            events_in_capacity: want_events_cap,
            cfg_buf,
            sync_sim_cfg_buf,
            sync_alive_bitmap_buf,
            drain_tail_staging,
            drain_records_staging,
            drain_ring_capacity: ring_cap,
        });
    }

    /// Process one batch of input events on GPU.
    ///
    /// Inputs:
    ///   * `agent_slots_in` — packed agent SoA (caller calls
    ///     `pack_agent_slots(state)`).
    ///   * `abilities` — packed ability registry; `empty()` if no
    ///     ability cascade fires this batch.
    ///   * `kin_lists` / `nearest_hostile` — per-agent pre-computed
    ///     spatial results. `kin_lists[slot]` is the `nearby_kin` result
    ///     for agent slot+1 at the driver-chosen radius (12 m for our
    ///     fear/pack/rally rules). `nearest_hostile[slot]` is the raw
    ///     AgentId of the closest hostile or `0xFFFFFFFFu`.
    ///   * `events_in` — the batch to dispatch physics on. Slots are
    ///     1:1 with `gpu_dispatch` invocations; one thread per event.
    ///   * `cfg` — kernel-local per-run config (`num_events`,
    ///     `agent_cap`, `max_abilities`, `max_effects`). World-scalars
    ///     (`tick`, `engagement_range`, `cascade_max_iterations`) flow
    ///     via `sim_cfg` — Task 2.8 of the GPU sim-state refactor.
    ///   * `sim_cfg` — shared world-scalar snapshot. The sync path
    ///     uploads it into the pool-owned `sync_sim_cfg_buf` each
    ///     dispatch so this caller doesn't need to manage a resident
    ///     handle; the resident path uses `run_batch_resident` which
    ///     takes a caller-supplied `sim_cfg_buf` instead.
    ///
    /// Outputs:
    ///   * `agent_slots_out` — mutated SoA after physics.
    ///   * `events_out` — events the kernel emitted this dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agent_slots_in: &[GpuAgentSlot],
        abilities: &PackedAbilityRegistry,
        kin_lists: &[GpuKinList],
        nearest_hostile: &[u32],
        events_in: &[EventRecord],
        cfg: PhysicsCfg,
        sim_cfg: &crate::sim_cfg::SimCfg,
    ) -> Result<PhysicsBatchOutput, PhysicsError> {
        let agent_cap = cfg.agent_cap;
        if (agent_slots_in.len() as u32) < agent_cap {
            return Err(PhysicsError::Dispatch(format!(
                "agent_slots_in len {} < cfg.agent_cap {}",
                agent_slots_in.len(),
                agent_cap
            )));
        }
        if (kin_lists.len() as u32) < agent_cap
            || (nearest_hostile.len() as u32) < agent_cap
        {
            return Err(PhysicsError::Dispatch(format!(
                "spatial results too small: kin_lists={} nearest_hostile={} agent_cap={}",
                kin_lists.len(),
                nearest_hostile.len(),
                agent_cap,
            )));
        }
        if events_in.is_empty() {
            // Nothing to dispatch — return a deep copy of the inputs.
            return Ok(PhysicsBatchOutput {
                agent_slots_out: agent_slots_in.to_vec(),
                events_out: Vec::new(),
                drain: DrainOutcome::default(),
            });
        }
        self.ensure_pool(device, agent_cap, events_in.len() as u32);
        let pool = self.pool.as_ref().expect("pool ensured");

        // Uploads.
        queue.write_buffer(
            &pool.agents_buf,
            0,
            bytemuck::cast_slice(&agent_slots_in[..agent_cap as usize]),
        );
        queue.write_buffer(
            &pool.abilities_known_buf,
            0,
            bytemuck::cast_slice(&abilities.known),
        );
        queue.write_buffer(
            &pool.abilities_cooldown_buf,
            0,
            bytemuck::cast_slice(&abilities.cooldown),
        );
        queue.write_buffer(
            &pool.abilities_effects_count_buf,
            0,
            bytemuck::cast_slice(&abilities.effects_count),
        );
        queue.write_buffer(
            &pool.abilities_effects_buf,
            0,
            bytemuck::cast_slice(&abilities.effects),
        );
        queue.write_buffer(
            &pool.kin_lists_buf,
            0,
            bytemuck::cast_slice(&kin_lists[..agent_cap as usize]),
        );
        queue.write_buffer(
            &pool.nearest_hostile_buf,
            0,
            bytemuck::cast_slice(&nearest_hostile[..agent_cap as usize]),
        );
        queue.write_buffer(&pool.events_in_buf, 0, bytemuck::cast_slice(events_in));

        let num_events = events_in.len() as u32;
        let cfg_on_wire = PhysicsCfg {
            num_events,
            ..cfg
        };
        queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&cfg_on_wire));
        // Task 2.8 — upload the SimCfg snapshot into the pool-owned
        // fallback buffer so the sync bind group's `sim_cfg` binding
        // sees current tick + engagement_range + cascade_max_iterations.
        crate::sim_cfg::upload_sim_cfg(queue, &pool.sync_sim_cfg_buf, sim_cfg);

        // Pack the alive bitmap host-side from `agent_slots_in` and
        // upload. Sync path has no `alive_pack_kernel` dispatch — we
        // already have the input slots on CPU so packing here is
        // cheaper than a GPU round-trip.
        {
            let words = crate::alive_bitmap::alive_bitmap_words(agent_cap) as usize;
            let mut packed = vec![0u32; words.max(1)];
            for (slot_idx, s) in agent_slots_in[..agent_cap as usize].iter().enumerate() {
                if s.alive != 0 {
                    packed[slot_idx >> 5] |= 1u32 << (slot_idx & 31);
                }
            }
            queue.write_buffer(&pool.sync_alive_bitmap_buf, 0, bytemuck::cast_slice(&packed));
        }

        // Reset the event ring tail so outputs start at slot 0.
        self.event_ring.reset(queue);

        // Build bind group — the event ring's buffers come from the
        // GpuEventRing (Arc-shared handles); everything else from our pool.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::physics::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.abilities_known_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pool.abilities_cooldown_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pool.abilities_effects_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pool.abilities_effects_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pool.kin_lists_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pool.nearest_hostile_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pool.events_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.event_ring.records_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.event_ring.tail_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: pool.cfg_buf.as_entire_binding(),
                },
                // Task 203 — chronicle ring bindings. The tail atomic
                // is NOT reset per dispatch: chronicle records
                // accumulate across ticks and are drained lazily via
                // `GpuBackend::flush_chronicle`.
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.chronicle_ring.records_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.chronicle_ring.tail_buffer().as_entire_binding(),
                },
                // Task 2.8 — shared SimCfg (sync path uses pool-owned
                // fallback buffer populated just above).
                wgpu::BindGroupEntry {
                    binding: SIM_CFG_BINDING,
                    resource: pool.sync_sim_cfg_buf.as_entire_binding(),
                },
                // Per-tick alive bitmap (sync path: host-packed above).
                wgpu::BindGroupEntry {
                    binding: crate::alive_bitmap::ALIVE_BITMAP_BINDING,
                    resource: pool.sync_alive_bitmap_buf.as_entire_binding(),
                },
            ],
        });

        // Phase 9 (task 195): encode physics dispatch + agents readback
        // copy + event-ring (tail + records) readback copy into ONE
        // command encoder so we submit + wait once per cascade iteration
        // rather than three times.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::physics::enc"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::physics::cpass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.pipeline);
            let groups = num_events.div_ceil(PHYSICS_WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        // Readback agents.
        encoder.copy_buffer_to_buffer(
            &pool.agents_buf,
            0,
            &pool.agents_readback,
            0,
            (agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64),
        );
        // Readback event-ring tail + records into the POOLED staging
        // (allocated once at pool init). Capacity is the ring's
        // capacity, so we always copy the full buffer; the drain logic
        // reads only `[0, tail_raw)`.
        if pool.drain_ring_capacity < self.event_ring_capacity {
            return Err(PhysicsError::Dispatch(format!(
                "drain staging sized for {} records but ring capacity is {}",
                pool.drain_ring_capacity, self.event_ring_capacity,
            )));
        }
        let ring_cap = self.event_ring_capacity;
        let ring_bytes = (ring_cap as u64) * (std::mem::size_of::<EventRecord>() as u64);
        encoder.copy_buffer_to_buffer(
            self.event_ring.tail_buffer(),
            0,
            &pool.drain_tail_staging,
            0,
            4,
        );
        encoder.copy_buffer_to_buffer(
            self.event_ring.records_buffer(),
            0,
            &pool.drain_records_staging,
            0,
            ring_bytes,
        );
        queue.submit(Some(encoder.finish()));

        // Issue all three map_async calls up front, then poll ONCE.
        // Three separate poll calls would serialise the waits; GPU can
        // finish all copies in parallel and the driver can wake all
        // three callbacks during a single poll.
        let agents_slice = pool.agents_readback.slice(..);
        let (agents_tx, agents_rx) = std::sync::mpsc::channel();
        agents_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = agents_tx.send(r);
        });
        let tail_slice = pool.drain_tail_staging.slice(..);
        let (tail_tx, tail_rx) = std::sync::mpsc::channel();
        tail_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tail_tx.send(r);
        });
        let records_slice = pool.drain_records_staging.slice(..);
        let (records_tx, records_rx) = std::sync::mpsc::channel();
        records_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = records_tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);

        // Collect agents.
        let agents_result = agents_rx.recv().map_err(|e| {
            PhysicsError::Dispatch(format!("agents readback channel closed: {e}"))
        })?;
        agents_result
            .map_err(|e| PhysicsError::Dispatch(format!("agents map_async: {e:?}")))?;
        let data = agents_slice.get_mapped_range();
        let agent_slots_out: Vec<GpuAgentSlot> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.agents_readback.unmap();

        // Collect tail.
        let tail_result = tail_rx
            .recv()
            .map_err(|e| PhysicsError::Dispatch(format!("tail map channel closed: {e}")))?;
        tail_result.map_err(|e| PhysicsError::Dispatch(format!("tail map: {e:?}")))?;
        let data = tail_slice.get_mapped_range();
        let mut tb = [0u8; 4];
        tb.copy_from_slice(&data[..4]);
        drop(data);
        pool.drain_tail_staging.unmap();
        let tail_raw = u32::from_le_bytes(tb);
        let drained_count = tail_raw.min(ring_cap);
        let overflowed = tail_raw > ring_cap;

        // Collect records. Always unmap even if we're not going to keep
        // them (drained_count == 0) so the next iteration's map_async
        // doesn't fail with "buffer already mapped".
        let records_result = records_rx
            .recv()
            .map_err(|e| PhysicsError::Dispatch(format!("records map channel closed: {e}")))?;
        records_result
            .map_err(|e| PhysicsError::Dispatch(format!("records map: {e:?}")))?;
        let events_out: Vec<EventRecord> = {
            let data = records_slice.get_mapped_range();
            let all: &[EventRecord] = bytemuck::cast_slice(&data);
            let out = if drained_count == 0 {
                Vec::new()
            } else {
                all[..drained_count as usize].to_vec()
            };
            drop(data);
            pool.drain_records_staging.unmap();
            out
        };
        let mut events_out = events_out;
        // Deterministic sort — matches GpuEventRing::drain's sort key.
        events_out.sort_by_key(|r| (r.tick, r.kind, r.payload[0]));

        Ok(PhysicsBatchOutput {
            agent_slots_out,
            events_out,
            drain: DrainOutcome {
                tail_raw,
                drained: drained_count,
                overflowed,
            },
        })
    }

    /// Task B7 — ensure the resident-path cfg-uniform pool is allocated
    /// and sized for `agent_cap`. The resident path's compute inputs
    /// (agents / abilities / kin / events / event_ring) all flow in as
    /// caller-supplied buffers, so this pool holds only the two
    /// uniforms the kernel still needs.
    fn ensure_resident_pool(&mut self, device: &wgpu::Device, agent_cap: u32) {
        let want_agent_cap = agent_cap.max(1);
        if let Some(p) = &self.pool_resident {
            if p.agent_cap == want_agent_cap {
                return;
            }
        }
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::physics::resident::cfg"),
            size: std::mem::size_of::<PhysicsCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One resident_cfg uniform per iteration slot, pre-populated
        // with `{read_slot: iter, write_slot: iter + 1}`. Contents are
        // static across all ticks so we write them once via
        // `mapped_at_creation` (no queue.write_buffer needed, no
        // collapse risk). See `ResidentBufferPool::resident_cfg_bufs`
        // docs for why we can't share a single buffer.
        let num_slots =
            (crate::cascade::MAX_CASCADE_ITERATIONS + 1) as usize;
        let mut resident_cfg_bufs: Vec<wgpu::Buffer> =
            Vec::with_capacity(num_slots);
        for iter in 0..num_slots {
            let cfg = ResidentPhysicsCfg {
                read_slot: iter as u32,
                // write_slot is one past read_slot. For the final iter
                // (read_slot = MAX_CASCADE_ITERATIONS) write_slot would
                // overflow the num_events / indirect_args buffers if
                // actually dispatched — but the cascade loop caps at
                // `iter < max_iters <= MAX_CASCADE_ITERATIONS`, so the
                // (read_slot = MAX) uniform is never bound. Still, we
                // allocate the slot so indexing by `read_slot = iter`
                // is uniform across the loop bounds.
                write_slot: (iter as u32).saturating_add(1),
                _pad0: 0,
                _pad1: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine_gpu::physics::resident::resident_cfg[i]"),
                size: std::mem::size_of::<ResidentPhysicsCfg>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::bytes_of(&cfg));
            }
            buf.unmap();
            resident_cfg_bufs.push(buf);
        }
        self.pool_resident = Some(ResidentBufferPool {
            agent_cap: want_agent_cap,
            cfg_buf,
            resident_cfg_bufs,
            last_cfg: None,
        });
        // Pool rebuilt — cached resident BGs reference the *old*
        // pool.cfg_buf / pool.resident_cfg_bufs, so drop them.
        // Repopulated lazily on next run_batch_resident call.
        self.resident_bg_cache.clear();
    }

    /// Task B7 — resident-path sibling to [`Self::run_batch`].
    ///
    /// Records ONE physics iteration dispatch into `encoder` using
    /// `dispatch_workgroups_indirect`, reading its workgroup count from
    /// `indirect_args.buffer()` at byte offset
    /// `indirect_args.slot_offset(read_slot)`. At end of dispatch, the
    /// kernel (thread 0) writes the workgroup count for iteration
    /// `write_slot` into `indirect_args[write_slot]` AND the event
    /// count into `num_events_buf[write_slot]`. When the kernel
    /// emitted zero events, the write lands `(0u, 1u, 1u)` — subsequent
    /// `dispatch_workgroups_indirect` calls become GPU no-ops,
    /// converging the cascade without a readback.
    ///
    /// The caller chains MAX_CASCADE_ITERATIONS calls with
    /// `(read_slot, write_slot)` pairs: iter 0 reads slot 0 (pre-seeded
    /// by the caller with the initial event count + workgroup count),
    /// writes slot 1; iter 1 reads slot 1, writes slot 2; and so on.
    /// Slot 0's seed is written by the driver (typically a tiny "seed"
    /// kernel that reads `apply_event_ring.tail` and writes
    /// `(ceil(tail/WG), 1, 1)` into slot 0 + `tail` into
    /// `num_events_buf[0]`).
    ///
    /// ### Scope (this signature intentionally differs from the plan)
    ///
    /// The plan sketch's signature collapses ability / kin buffers into
    /// single handles; the implementation splits them back out to match
    /// the sync path's bind-group layout (4 ability buffers, 1 kin
    /// buffer, 1 nearest-hostile buffer). This keeps the WGSL shared
    /// between sync + resident and avoids the caller having to re-pack
    /// data into an aliased layout. It also adds `events_in_buf`
    /// explicitly — the WGSL binding 7 needs a read-only buffer of
    /// records, and wgpu forbids aliasing the same buffer as both
    /// read-only and read-write in the same bind group, so events_in
    /// cannot be satisfied from `event_ring.records_buffer()` alone.
    /// The C1/C2 driver (future work) will own the ping-pong between
    /// the input events buffer and the output event ring.
    ///
    /// ### Preconditions
    ///
    /// * `agents_buf` is at least `agent_cap * size_of::<GpuAgentSlot>()`
    ///   bytes, STORAGE (read_write).
    /// * `abilities_*_buf` match the sync `PackedAbilityRegistry` sizes.
    /// * `kin_buf` / `nearest_hostile_buf` are sized per `agent_cap`.
    /// * `events_in_buf` has at least `ceil(num_events/WG)*WG` records
    ///   — beyond the current `num_events`, contents are ignored.
    /// * `event_ring` is caller-owned; its tail should be reset by the
    ///   caller before iter 0, then left alone (physics appends via
    ///   atomicAdd).
    /// * `indirect_args` has at least `write_slot+1` slots.
    /// * `num_events_buf_size_u32 >= write_slot+1`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        abilities_known_buf: &wgpu::Buffer,
        abilities_cooldown_buf: &wgpu::Buffer,
        abilities_effects_count_buf: &wgpu::Buffer,
        abilities_effects_buf: &wgpu::Buffer,
        kin_buf: &wgpu::Buffer,
        nearest_hostile_buf: &wgpu::Buffer,
        events_in_buf: &wgpu::Buffer,
        event_ring: &GpuEventRing,
        chronicle_ring: &GpuChronicleRing,
        indirect_args: &crate::gpu_util::indirect::IndirectArgsBuffer,
        num_events_buf: &wgpu::Buffer,
        sim_cfg_buf: &wgpu::Buffer,
        gold_buf: &wgpu::Buffer,
        standing_records_buf: &wgpu::Buffer,
        standing_counts_buf: &wgpu::Buffer,
        memory_records_buf: &wgpu::Buffer,
        memory_cursors_buf: &wgpu::Buffer,
        alive_bitmap_buf: &wgpu::Buffer,
        read_slot: u32,
        write_slot: u32,
        cfg: PhysicsCfg,
    ) -> Result<(), PhysicsError> {
        if read_slot >= indirect_args.slots() || write_slot >= indirect_args.slots() {
            return Err(PhysicsError::Dispatch(format!(
                "indirect slot OOB: read={read_slot} write={write_slot} slots={}",
                indirect_args.slots()
            )));
        }
        let agent_cap = cfg.agent_cap;
        self.ensure_resident_pool(device, agent_cap);

        // cfg uniform dedupe: the same cfg is used for all 8 cascade
        // iters of a tick; only `tick` changes between ticks. Dedupe
        // saves 7/8 queue.write_buffer calls per tick.
        {
            let pool_mut = self.pool_resident.as_mut().expect("resident pool ensured");
            if pool_mut.last_cfg != Some(cfg) {
                queue.write_buffer(&pool_mut.cfg_buf, 0, bytemuck::bytes_of(&cfg));
                pool_mut.last_cfg = Some(cfg);
            }
        }
        let pool = self
            .pool_resident
            .as_ref()
            .expect("resident pool ensured");
        // Sanity: write_slot must be read_slot + 1 — the pool
        // pre-populated `resident_cfg_bufs[i]` with
        // `{read_slot: i, write_slot: i+1}`, so off-by-one callers
        // would silently bind the wrong iter's uniform.
        debug_assert_eq!(
            write_slot,
            read_slot.saturating_add(1),
            "run_batch_resident: write_slot must equal read_slot+1; got \
             read_slot={read_slot} write_slot={write_slot}"
        );
        let resident_cfg_buf = pool
            .resident_cfg_bufs
            .get(read_slot as usize)
            .ok_or_else(|| {
                PhysicsError::Dispatch(format!(
                    "read_slot {} exceeds pre-allocated resident_cfg_bufs slots ({})",
                    read_slot,
                    pool.resident_cfg_bufs.len()
                ))
            })?
            .clone();

        // Cache the bind group. Across 8 cascade iterations per tick
        // the caller drives us with at most 3 distinct buffer tuples
        // (iter 0: apply-ring in, ring_a out; odd: ring_a in, ring_b
        // out; even ≥ 2: ring_b in, ring_a out). All other bindings
        // (agents, abilities, spatial, indirect_args, etc.) are stable
        // across a batch, so the cache hits 100% after the first tick.
        let key = ResidentBgKey {
            agent_cap,
            agents: agents_buf.clone(),
            ability_known: abilities_known_buf.clone(),
            ability_cooldown: abilities_cooldown_buf.clone(),
            ability_effects_count: abilities_effects_count_buf.clone(),
            ability_effects: abilities_effects_buf.clone(),
            kin: kin_buf.clone(),
            nearest_hostile: nearest_hostile_buf.clone(),
            events_in: events_in_buf.clone(),
            event_ring_records: event_ring.records_buffer().clone(),
            event_ring_tail: event_ring.tail_buffer().clone(),
            chronicle_records: chronicle_ring.records_buffer().clone(),
            chronicle_tail: chronicle_ring.tail_buffer().clone(),
            indirect_args: indirect_args.buffer().clone(),
            num_events: num_events_buf.clone(),
            sim_cfg: sim_cfg_buf.clone(),
            gold_buf: gold_buf.clone(),
            // Per-iteration resident_cfg uniform — distinct identity
            // per iter so each iter gets its own cached BG rather than
            // aliasing the previous iter's (which would silently bind
            // the wrong read_slot). — Task #68 fix.
            resident_cfg: resident_cfg_buf.clone(),
            // Task #79 SP-4 — standing view storage (records + counts).
            standing_records: standing_records_buf.clone(),
            standing_counts: standing_counts_buf.clone(),
            // Subsystem 2 Phase 4 PR-4 — memory view storage
            // (records + cursors).
            memory_records: memory_records_buf.clone(),
            memory_cursors: memory_cursors_buf.clone(),
            // Per-tick alive bitmap (resident path: packed by
            // `alive_pack_kernel` at the top of each tick).
            alive_bitmap: alive_bitmap_buf.clone(),
        };
        // Perf Stage A.2 — probe-before-insert so the hit/miss
        // counters track actual cache behaviour. `HashMap::entry`
        // doesn't expose whether it was `Occupied` vs `Vacant` without
        // moving the key, so we do a pre-lookup. Cost is one extra
        // hash + buffer-ptr comparison per dispatch (cheap — the whole
        // point of this cache is that key hashing is fast).
        if self.resident_bg_cache.contains_key(&key) {
            self.resident_bg_cache_hits = self.resident_bg_cache_hits.saturating_add(1);
        } else {
            self.resident_bg_cache_misses = self.resident_bg_cache_misses.saturating_add(1);
        }
        let bind_group: &wgpu::BindGroup = self.resident_bg_cache.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::physics::bg_resident"),
                layout: &self.bind_group_layout_resident,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agents_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: abilities_known_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: abilities_cooldown_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: abilities_effects_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: abilities_effects_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: kin_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: nearest_hostile_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: events_in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: event_ring.records_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: event_ring.tail_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: pool.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: chronicle_ring.records_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: chronicle_ring.tail_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: indirect_args.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: num_events_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: resident_cfg_buf.as_entire_binding(),
                    },
                    // Task 2.8 — caller-supplied shared SimCfg buffer.
                    wgpu::BindGroupEntry {
                        binding: SIM_CFG_BINDING,
                        resource: sim_cfg_buf.as_entire_binding(),
                    },
                    // Phase 3 Task 3.4 — gold ledger side buffer.
                    wgpu::BindGroupEntry {
                        binding: 17,
                        resource: gold_buf.as_entire_binding(),
                    },
                    // Task #79 SP-4 — standing view storage.
                    wgpu::BindGroupEntry {
                        binding: 18,
                        resource: standing_records_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 19,
                        resource: standing_counts_buf.as_entire_binding(),
                    },
                    // Subsystem 2 Phase 4 PR-4 — memory view storage.
                    wgpu::BindGroupEntry {
                        binding: 20,
                        resource: memory_records_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 21,
                        resource: memory_cursors_buf.as_entire_binding(),
                    },
                    // Per-tick alive bitmap (resident path: packed by
                    // `alive_pack_kernel` at the top of each tick).
                    wgpu::BindGroupEntry {
                        binding: crate::alive_bitmap::ALIVE_BITMAP_BINDING,
                        resource: alive_bitmap_buf.as_entire_binding(),
                    },
                ],
            })
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::physics::cpass_resident"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_resident);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups_indirect(
                indirect_args.buffer(),
                indirect_args.slot_offset(read_slot),
            );
        }
        Ok(())
    }
}

// Phase 9 (task 195): `drain_raw_records` retired. The fused
// submit+readback path in `PhysicsKernel::run_batch` inlines the tail
// + records copy into the same encoder as the physics dispatch and
// the agents-readback copy, eliminating two extra submits + two extra
// map_async waits per cascade iteration.

/// Result of a single `PhysicsKernel::run_batch` call.
pub struct PhysicsBatchOutput {
    /// Agent SoA after the dispatch — apply to SimState via
    /// `unpack_agent_slots`.
    pub agent_slots_out: Vec<GpuAgentSlot>,
    /// Raw records emitted by physics this dispatch, sorted by
    /// (tick, kind, payload[0]) for determinism.
    pub events_out: Vec<EventRecord>,
    /// Drain outcome — `overflowed == true` is a hard failure (events
    /// were dropped).
    pub drain: DrainOutcome,
}

/// Helper: unpack raw records into CPU `Event`s + push into an EventRing.
pub fn events_into_ring(records: &[EventRecord], ring: &mut EventRing) -> usize {
    let mut pushed = 0;
    for r in records {
        if let Some(e) = unpack_record(r) {
            ring.push(e);
            pushed += 1;
        }
    }
    pushed
}

/// Helper: pack a CPU `EventRing` batch into `Vec<EventRecord>` for the
/// kernel's `events_in` binding. Events that aren't packable
/// (e.g. `ChronicleEntry`) are filtered out.
pub fn pack_events_for_kernel(events: &[Event]) -> Vec<EventRecord> {
    events.iter().filter_map(pack_event).collect()
}

// ---------------------------------------------------------------------------
// Silence `PAYLOAD_WORDS` unused warning when tests don't reach for it.
// ---------------------------------------------------------------------------
#[allow(dead_code)]
const _PAYLOAD_WORDS_ASSERT: usize = PAYLOAD_WORDS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_agent_slot_size_is_64_bytes() {
        // 16 u32-sized fields → 64 bytes. WGSL struct alignment matches
        // when we keep the trailing `_pad0/_pad1`.
        assert_eq!(std::mem::size_of::<GpuAgentSlot>(), 64);
    }

    #[test]
    fn effect_op_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<GpuEffectOp>(), 16);
    }

    #[test]
    fn physics_cfg_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<PhysicsCfg>(), 32);
    }

    #[test]
    fn resident_physics_cfg_size_is_16_bytes() {
        // WGSL `ResidentPhysicsCfg { read_slot, write_slot, _pad0,
        // _pad1 }` must match host size so the uniform upload has the
        // exact bytes the kernel expects.
        assert_eq!(std::mem::size_of::<ResidentPhysicsCfg>(), 16);
    }

    #[test]
    fn physics_shader_parses_through_naga() {
        // Assemble the full WGSL shader from every rule in physics.sim
        // and feed it through naga — catches integration bugs (missing
        // stub fns, stale event-kind consts, struct drift) without
        // requiring a GPU device.
        use dsl_compiler::ast::Program;
        use std::fs;
        use std::path::PathBuf;

        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop(); // crates/
        root.pop(); // repo root
        root.push("assets/sim");

        let mut merged = Program { decls: Vec::new() };
        for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
            let src = fs::read_to_string(root.join(f)).expect("read sim source");
            merged.decls.extend(dsl_compiler::parse(&src).expect("parse").decls);
        }
        let comp = dsl_compiler::compile_ast(merged).expect("resolve");

        let ctx = EmitContext {
            events: &comp.events,
            event_tags: &comp.event_tags,
        };
        let wgsl = build_physics_shader(&comp.physics, &ctx, 1024)
            .expect("build shader");

        if let Err(e) = naga::front::wgsl::parse_str(&wgsl) {
            panic!(
                "physics shader failed naga parse:\n{e}\n--- WGSL source ---\n{wgsl}"
            );
        }
    }

    #[test]
    fn physics_resident_shader_parses_through_naga() {
        // Task B7 — mirror of `physics_shader_parses_through_naga` for
        // the resident shader. Catches WGSL bugs in the appended
        // bindings + `cs_physics_resident` entry point without
        // requiring a GPU device.
        use dsl_compiler::ast::Program;
        use std::fs;
        use std::path::PathBuf;

        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop();
        root.pop();
        root.push("assets/sim");

        let mut merged = Program { decls: Vec::new() };
        for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
            let src = fs::read_to_string(root.join(f)).expect("read sim source");
            merged.decls.extend(dsl_compiler::parse(&src).expect("parse").decls);
        }
        let comp = dsl_compiler::compile_ast(merged).expect("resolve");

        let ctx = EmitContext {
            events: &comp.events,
            event_tags: &comp.event_tags,
        };
        let wgsl = build_physics_shader_resident(
            &comp.physics,
            &ctx,
            1024,
            DEFAULT_CHRONICLE_CAPACITY,
        )
        .expect("build resident shader");

        if let Err(e) = naga::front::wgsl::parse_str(&wgsl) {
            panic!(
                "physics resident shader failed naga parse:\n{e}\n--- WGSL source ---\n{wgsl}"
            );
        }

        // Sanity check: the resident entry point is present.
        assert!(
            wgsl.contains("fn cs_physics_resident"),
            "resident entry point missing"
        );
        // And the new bindings are declared.
        assert!(wgsl.contains("@binding(13) var<storage, read_write> indirect_args"));
        assert!(wgsl.contains("@binding(14) var<storage, read_write> num_events_buf"));
        assert!(wgsl.contains("@binding(15) var<uniform> resident_cfg"));
    }

    #[test]
    fn pack_agent_slots_roundtrips_alive() {
        use engine_data::entities::CreatureType;
        use engine::state::AgentSpawn;
        use glam::Vec3;

        let mut state = SimState::new(4, 0xDEAD_BEEF);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(1.0, 2.0, 3.0),
                hp: 42.0,
                ..Default::default()
            })
            .expect("spawn");
        let slots = pack_agent_slots(&state);
        assert_eq!(slots.len(), 4);
        assert_eq!(slots[0].alive, 1);
        assert_eq!(slots[0].hp, 42.0);
        assert_eq!(slots[0].pos_x, 1.0);
        assert_eq!(slots[0].creature_type, CreatureType::Human as u32);
        assert_eq!(slots[0].engaged_with, GpuAgentSlot::ENGAGED_NONE);
        // Unspawned slots
        assert_eq!(slots[1].alive, 0);
    }

    // -----------------------------------------------------------------------
    // PackedAbilityRegistry — Phase 1 tag surface (GPU ability-evaluation)
    //
    // The tag surface is unbound in Phase 1; these tests pin the packer's
    // wire shape so Phase 4's WGSL binding can assume `hints` /
    // `tag_values` are well-formed.
    // -----------------------------------------------------------------------

    #[test]
    fn packed_ability_registry_empty_shape_includes_tag_surface() {
        let packed = PackedAbilityRegistry::empty();
        assert_eq!(packed.known.len(), MAX_ABILITIES);
        assert_eq!(packed.cooldown.len(), MAX_ABILITIES);
        assert_eq!(packed.effects_count.len(), MAX_ABILITIES);
        assert_eq!(packed.effects.len(), MAX_ABILITIES * MAX_EFFECTS);

        // Phase 1 additions: hint sentinel per slot + tag row per slot.
        assert_eq!(packed.hints.len(), MAX_ABILITIES);
        assert!(
            packed.hints.iter().all(|h| *h == HINT_NONE_SENTINEL),
            "empty() must mark every slot as HINT_NONE_SENTINEL"
        );
        assert_eq!(
            packed.tag_values.len(),
            MAX_ABILITIES * engine::ability::AbilityTag::COUNT,
        );
        assert!(
            packed.tag_values.iter().all(|v| *v == 0.0),
            "empty() must zero every tag entry"
        );
    }

    #[test]
    fn pack_carries_hint_and_tag_values_through_lowering() {
        use engine::ability::{
            AbilityHint, AbilityProgram, AbilityRegistryBuilder, AbilityTag, EffectOp, Gate,
        };

        let mut b = AbilityRegistryBuilder::new();

        // Slot 0 — damage hint + mixed tag vector.
        let a0 = AbilityProgram::new_single_target(
            6.0,
            Gate { cooldown_ticks: 40, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        )
        .with_hint(AbilityHint::Damage)
        .with_tags([
            (AbilityTag::Physical, 80.0),
            (AbilityTag::CrowdControl, 10.0),
        ]);
        let _id_a = b.register(a0);

        // Slot 1 — no hint, single tag.
        let a1 = AbilityProgram::new_single_target(
            4.0,
            Gate { cooldown_ticks: 20, hostile_only: false, line_of_sight: false },
            [EffectOp::Heal { amount: 25.0 }],
        )
        .with_tags([(AbilityTag::Heal, 60.0)]);
        let _id_b = b.register(a1);

        // Slot 2 — defense hint, no tags (tag reads default to 0).
        let a2 = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 100, hostile_only: false, line_of_sight: false },
            [EffectOp::Shield { amount: 30.0 }],
        )
        .with_hint(AbilityHint::Defense);
        let _id_c = b.register(a2);

        let registry = b.build();
        let packed = PackedAbilityRegistry::pack(&registry);

        // Shape preserved.
        assert_eq!(packed.hints.len(), MAX_ABILITIES);
        assert_eq!(
            packed.tag_values.len(),
            MAX_ABILITIES * AbilityTag::COUNT,
        );

        // Slot 0: damage hint, PHYSICAL=80 + CROWD_CONTROL=10.
        assert_eq!(packed.known[0], 1);
        assert_eq!(packed.cooldown[0], 40);
        assert_eq!(packed.effects_count[0], 1);
        assert_eq!(packed.hints[0], AbilityHint::Damage.discriminant() as u32);
        let stride = AbilityTag::COUNT;
        assert_eq!(packed.tag_values[0 * stride + AbilityTag::Physical.index()], 80.0);
        assert_eq!(
            packed.tag_values[0 * stride + AbilityTag::CrowdControl.index()],
            10.0,
        );
        assert_eq!(packed.tag_values[0 * stride + AbilityTag::Heal.index()], 0.0);

        // Slot 1: no hint → sentinel; HEAL=60.
        assert_eq!(packed.known[1], 1);
        assert_eq!(packed.hints[1], HINT_NONE_SENTINEL);
        assert_eq!(packed.tag_values[1 * stride + AbilityTag::Heal.index()], 60.0);
        assert_eq!(
            packed.tag_values[1 * stride + AbilityTag::Physical.index()],
            0.0,
        );

        // Slot 2: defense hint, all tag values zero.
        assert_eq!(packed.known[2], 1);
        assert_eq!(packed.hints[2], AbilityHint::Defense.discriminant() as u32);
        for t in AbilityTag::all() {
            assert_eq!(
                packed.tag_values[2 * stride + t.index()],
                0.0,
                "slot 2 tag {:?} should be 0",
                t,
            );
        }

        // Slot 3+ unset → sentinel + zeros.
        assert_eq!(packed.known[3], 0);
        assert_eq!(packed.hints[3], HINT_NONE_SENTINEL);
    }

    #[test]
    fn packer_truncates_beyond_max_abilities() {
        // If someone registers more than MAX_ABILITIES programs, the
        // packer drops the overflow rather than panicking. Documented
        // in `PackedAbilityRegistry::pack`.
        use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};

        let mut b = AbilityRegistryBuilder::new();
        // Register MAX_ABILITIES + 2 programs.
        for _ in 0..(MAX_ABILITIES + 2) {
            b.register(AbilityProgram::new_single_target(
                1.0,
                Gate { cooldown_ticks: 1, hostile_only: false, line_of_sight: false },
                [EffectOp::Damage { amount: 1.0 }],
            ));
        }
        let registry = b.build();
        let packed = PackedAbilityRegistry::pack(&registry);

        // Every budgeted slot is populated.
        assert_eq!(packed.known.len(), MAX_ABILITIES);
        assert!(packed.known.iter().all(|k| *k == 1));
        // Overflow slots simply don't exist in the packed buffer.
    }

    #[test]
    fn packer_preserves_effects_up_to_cpu_cap() {
        use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};

        // The CPU cap `MAX_EFFECTS_PER_PROGRAM` (4) is always `<=` the
        // GPU cap `MAX_EFFECTS` (8), so a fully-packed CPU program
        // round-trips without truncation. This pins that relationship
        // — if the CPU cap is ever raised past MAX_EFFECTS the packer
        // loop's `.take(MAX_EFFECTS)` becomes load-bearing.
        let effects: Vec<EffectOp> = (0..engine::ability::MAX_EFFECTS_PER_PROGRAM)
            .map(|_| EffectOp::Damage { amount: 1.0 })
            .collect();

        let mut b = AbilityRegistryBuilder::new();
        b.register(AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 1, hostile_only: false, line_of_sight: false },
            effects,
        ));
        let registry = b.build();
        let packed = PackedAbilityRegistry::pack(&registry);

        assert!(
            engine::ability::MAX_EFFECTS_PER_PROGRAM <= MAX_EFFECTS,
            "MAX_EFFECTS_PER_PROGRAM ({}) must fit in GPU MAX_EFFECTS ({})",
            engine::ability::MAX_EFFECTS_PER_PROGRAM,
            MAX_EFFECTS,
        );
        assert_eq!(
            packed.effects_count[0] as usize,
            engine::ability::MAX_EFFECTS_PER_PROGRAM,
        );
        for i in 0..engine::ability::MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(packed.effects[i].kind, 0, "slot 0 eff {} kind", i);
        }
    }
}
