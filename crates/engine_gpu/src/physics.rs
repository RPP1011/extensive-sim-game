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
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::SimState;
use crate::event_ring::{
    chronicle_wgsl_prefix, pack_event, unpack_record, wgsl_prefix, DrainOutcome, EventRecord,
    GpuChronicleRing, GpuEventRing, CHRONICLE_RING_WGSL, DEFAULT_CHRONICLE_CAPACITY,
    EVENT_RING_WGSL, PAYLOAD_WORDS,
};

/// Workgroup size for the physics dispatcher.
pub const PHYSICS_WORKGROUP_SIZE: u32 = 64;

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

/// Config uniform the physics kernel reads.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct PhysicsCfg {
    pub tick: u32,
    pub num_events: u32,
    pub combat_engagement_range: f32,
    pub cascade_max_iterations: u32,
    pub agent_cap: u32,
    pub max_abilities: u32,
    pub max_effects: u32,
    pub _pad: u32,
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

/// Flat ability registry ready for upload.
pub struct PackedAbilityRegistry {
    pub known: Vec<u32>,
    pub cooldown: Vec<u32>,
    pub effects_count: Vec<u32>,
    /// `effects[ab * MAX_EFFECTS + effect_idx]`.
    pub effects: Vec<GpuEffectOp>,
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
        }
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
         \x20 tick: u32,\n\
         \x20 num_events: u32,\n\
         \x20 combat_engagement_range: f32,\n\
         \x20 cascade_max_iterations: u32,\n\
         \x20 agent_cap: u32,\n\
         \x20 max_abilities: u32,\n\
         \x20 max_effects: u32,\n\
         \x20 _pad: u32,\n\
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

    // ---- State stub fns ----
    //
    // Every stub writes/reads exactly one field of the `agents` buffer.
    // Stubs that the CPU side doesn't have a GPU-side source for (gold,
    // standing, memory) are no-ops documented as such.
    out.push_str(&state_stub_fns());

    // ---- `cfg` namespace-field lookups (via `cfg.<snake>` in emitter) ----
    //
    // The emitter lowers `config.combat.engagement_range` to
    // `cfg.combat_engagement_range` and `cascade.max_iterations` to
    // `cfg.cascade_max_iterations`. Our `PhysicsConfig` struct uses
    // those exact field names, so the reads land correctly.

    // ---- `wgsl_world_tick` (tick uniform) ----
    out.push_str(
        "fn wgsl_world_tick_fn() -> u32 { return cfg.tick; }\n\
         // Shadow variable for the emitter's implicit tick reference —\n\
         // the emitted code spells it `wgsl_world_tick` as a bare name,\n\
         // so an init-at-top-of-each-fn alias is impractical. Rely on a\n\
         // module-level `let`-equivalent via a const expression fn the\n\
         // lowered code calls instead.\n",
    );
    // Actually we need `wgsl_world_tick` as an identifier — emit it as a
    // local shadowed in the dispatcher. The emitter doesn't generate fn
    // bodies that initialise this local; it just references
    // `wgsl_world_tick` as if it's in scope. We declare it as a
    // module-level `const` by embedding cfg.tick inline via a helper,
    // but WGSL module-level consts can't reference uniform storage.
    //
    // Workaround: emit a per-rule wrapper that injects
    // `let wgsl_world_tick = cfg.tick;` at the top. That keeps the
    // emitter output unchanged and scopes the variable per-fn.
    //
    // Implementation: rather than rewriting the emitter, wrap each
    // rule fn in a macro-expand step. Simpler: declare the ident as a
    // function-scope `let` inside every emitted fn. The emitter already
    // emits `let ... = ...` lines at the top of each rule body from the
    // destructure prelude, so we post-process the emitted source by
    // injecting one more `let` line right after the `fn ... {` header.

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

/// Inject `let wgsl_world_tick = cfg.tick;` right after the rule's
/// `fn name(event_idx: u32) {` header. The emitter produces that
/// identifier as a bare name in emit sites; declaring it as a function-
/// scope `let` keeps the emitter output unchanged.
fn wrap_rule_with_tick_alias(wgsl: &str) -> String {
    // Find the first `{` after `fn physics_...(event_idx: u32)`.
    let mut out = String::with_capacity(wgsl.len() + 64);
    let mut injected = false;
    for line in wgsl.lines() {
        out.push_str(line);
        out.push('\n');
        if !injected && line.contains("fn physics_") && line.trim_end().ends_with('{') {
            out.push_str("    let wgsl_world_tick: u32 = cfg.tick;\n");
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
fn state_stub_fns() -> String {
    // The emitter lists every stub name in the module doc. All scalar
    // getters/setters project into a single field of `agents[id - 1]`
    // (ids are 1-based; slot 0 is unused).
    //
    // `id` is bounds-checked — out-of-range ids return sentinel / are
    // no-ops so a malformed event can't corrupt state.
    r#"
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
// Gold mutations: no-op on GPU. The ModifyStanding / gold transfer
// rules still fire on the CPU side; the GPU path preserves state by
// leaving the (absent) gold buffer untouched.
fn state_set_agent_gold(id: u32, v: i32) { }
fn state_add_agent_gold(id: u32, delta: i32) { }
fn state_adjust_standing(a: u32, b: u32, delta: i32) { }
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
// Memory push: no-op. The CPU cascade owns the memory ring; the GPU
// record_memory rule lands here and documents the gap. If a future
// phase wants GPU memory, provision a ring buffer + atomic tail and
// replace this no-op.
fn state_push_agent_memory(observer: u32, source: u32, payload: u32, confidence: f32, t: u32) { }
"#.to_string()
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

        let event_ring = GpuEventRing::new(device, event_ring_capacity);
        let chronicle_ring = GpuChronicleRing::new(device, chronicle_ring_capacity);

        Ok(Self {
            pipeline,
            bind_group_layout,
            event_ring,
            chronicle_ring,
            pool: None,
            event_ring_capacity,
            chronicle_ring_capacity,
        })
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
    ///   * `cfg` — per-run config (tick, agent_cap, etc.).
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
    fn pack_agent_slots_roundtrips_alive() {
        use engine::creature::CreatureType;
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
}
