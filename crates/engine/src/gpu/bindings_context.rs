//! `KernelBindingsContext` — the shared bundle of buffer sources the
//! compiler-emitted `Bindings::from_context()` constructors pull from.
//!
//! # Why this exists
//!
//! Today every per-runtime `step()` writes ~30-line `Bindings { ... }`
//! literals by hand for every kernel dispatch — repeating the
//! `event_ring: self.event_ring.ring(), event_tail: self.event_ring.tail(),
//! agent_pos: &self.agent_pos_buf, ...` plumbing across ~40 runtime
//! crates. Every new SoA column or compiler-emitted buffer fans out
//! through every dispatch site.
//!
//! The dsl_compiler already emits the `Bindings` struct definitions; it
//! also emits a `from_context(&KernelBindingsContext)` constructor (and
//! a sibling `from_context_with_extras(...)` for kernels that bind
//! fixture-specific buffers) that pulls each buffer from the appropriate
//! shared source by naming convention. Per-runtime dispatch becomes:
//!
//! ```ignore
//! let ctx = KernelBindingsContext { state: &self.agent_buffers, event_ring: &self.event_ring, registry: &self.registry_gpu };
//! let bindings = SomeKernelBindings::from_context(&ctx);
//! ```
//!
//! # Naming convention (the constructor's getter table)
//!
//! The compiler classifies each emitted Bindings field name and routes
//! it via:
//!
//! | Field name pattern              | Source           | Rust expression                          |
//! |---------------------------------|------------------|------------------------------------------|
//! | `event_ring`                    | `ctx.event_ring` | `ctx.event_ring.ring()`                  |
//! | `event_tail`                    | `ctx.event_ring` | `ctx.event_ring.tail()`                  |
//! | `agent_<col>` (standard SoA)    | `ctx.state`      | `ctx.state.<col>_buf`                    |
//! | `ability_registry_<col>`        | `ctx.registry`   | `&ctx.registry.<col>`                    |
//! | `voxel_grid` (Phase C)          | `ctx.voxel_grid` | `ctx.voxel_grid.expect(...)`             |
//! | (anything else)                 | `extras`         | `extras.<name>`                          |
//!
//! The "anything else" bucket catches per-fixture mask bitmaps,
//! fixture-specific SoA columns (e.g. `agent_creature_type` in
//! spy_network), per-kernel cfg uniforms, scoring output buffers — all
//! the buffers that aren't shared infrastructure. These ride a
//! per-kernel `<KernelName>Extras` struct supplied as a trailing arg.
//!
//! # Phase 1 scope (this file)
//!
//! Phase 1 defines the context + the [`AgentBuffers`] aggregate the
//! constructors pull standard agent-SoA buffers from. No runtime is
//! migrated this slice — per-runtime hand-written `Bindings { ... }`
//! literals continue to work alongside the new constructors. Migrations
//! land in subsequent slices.
//!
//! # Why `AgentBuffers` instead of `&SimState`
//!
//! `SimState` (in [`crate::state`]) owns the host-side per-agent SoA
//! (`Vec<f32>`, `Vec<u32>`, etc.) used by host-driven helpers. It does
//! not own GPU buffer handles today — those live on per-runtime structs
//! (e.g. `SpyNetworkState::agent_hp_buf`). Rather than restructure
//! `SimState` to own `wgpu::Buffer`s as a precondition for this slice,
//! [`AgentBuffers`] is a thin reference aggregate per-runtime structs
//! populate at dispatch time. Phase 2 migrations will construct it
//! inline when calling `from_context()`. If a future architectural
//! change moves buffer ownership onto `SimState`, [`AgentBuffers`] can
//! become a view returned by a `SimState` accessor without disturbing
//! the constructor surface.

/// Per-agent SoA buffer references the standard naming convention
/// resolves through [`KernelBindingsContext::state`].
///
/// Every field corresponds to a known per-agent column the compiler
/// emits as an `agent_<col>` binding. Fixture-specific columns
/// (e.g. `agent_creature_type` in spy_network, `agent_disguise_*` in
/// spy_network) are NOT here — they go through the per-kernel
/// `<KernelName>Extras` struct instead.
///
/// Per-runtime construction at dispatch time:
///
/// ```ignore
/// let agent_buffers = engine::gpu::AgentBuffers {
///     hp_buf: Some(&self.agent_hp_buf),
///     max_hp_buf: Some(&self.agent_max_hp_buf),
///     // ... only the columns this runtime actually owns
///     ..Default::default()
/// };
/// ```
///
/// Each field is `Option<&'a wgpu::Buffer>` so a runtime that doesn't
/// own a particular column (e.g. a stress fixture with no mana stat)
/// can leave it `None`. The compiler-emitted `from_context()` body
/// `.expect("...")` unwraps the columns its kernel actually binds at
/// dispatch time, surfacing missing buffers as a clear runtime error
/// (`"kernel binds agent_mana but the runtime didn't supply
/// ctx.state.mana_buf"`) rather than a confusing borrow-of-None.
#[derive(Default)]
pub struct AgentBuffers<'a> {
    pub hp_buf: Option<&'a wgpu::Buffer>,
    pub max_hp_buf: Option<&'a wgpu::Buffer>,
    pub alive_buf: Option<&'a wgpu::Buffer>,
    pub pos_buf: Option<&'a wgpu::Buffer>,
    pub level_buf: Option<&'a wgpu::Buffer>,
    pub move_speed_buf: Option<&'a wgpu::Buffer>,
    pub move_speed_mult_buf: Option<&'a wgpu::Buffer>,
    pub shield_hp_buf: Option<&'a wgpu::Buffer>,
    pub armor_buf: Option<&'a wgpu::Buffer>,
    pub magic_resist_buf: Option<&'a wgpu::Buffer>,
    pub attack_damage_buf: Option<&'a wgpu::Buffer>,
    pub attack_range_buf: Option<&'a wgpu::Buffer>,
    pub mana_buf: Option<&'a wgpu::Buffer>,
    pub max_mana_buf: Option<&'a wgpu::Buffer>,
    pub ability_power_buf: Option<&'a wgpu::Buffer>,
}

impl<'a> AgentBuffers<'a> {
    /// Standard agent SoA column names the compiler resolves as
    /// `agent_<col>` → `ctx.state.<col>_buf`. The compiler matches an
    /// emitted Bindings field name `agent_<col>` against this list to
    /// decide whether the binding routes through `ctx.state` (returns
    /// `true`) or falls through to extras (returns `false`).
    ///
    /// Mirrors the field set above. Adding a new standard SoA column
    /// requires editing both the field list and this table — kept
    /// adjacent so drift is mechanically obvious.
    pub const STANDARD_COLUMNS: &'static [&'static str] = &[
        "hp",
        "max_hp",
        "alive",
        "pos",
        "level",
        "move_speed",
        "move_speed_mult",
        "shield_hp",
        "armor",
        "magic_resist",
        "attack_damage",
        "attack_range",
        "mana",
        "max_mana",
        "ability_power",
    ];

    /// Returns `true` iff `col` (the suffix after `agent_`) is a
    /// standard SoA column the compiler should route through `ctx.state`.
    pub fn is_standard_column(col: &str) -> bool {
        Self::STANDARD_COLUMNS.contains(&col)
    }
}

/// The shared bundle of buffer sources the compiler-emitted
/// `Bindings::from_context()` constructors pull from.
///
/// See module docs for the full naming-convention table. Per-runtime
/// `step()` builds one of these per dispatch (or once per tick if all
/// dispatches share the same context) and passes it to each kernel's
/// generated constructor.
pub struct KernelBindingsContext<'a> {
    /// Per-agent SoA buffers — the `agent_<col>` family of bindings.
    pub state: &'a AgentBuffers<'a>,
    /// Per-fixture event ring infrastructure — `event_ring` /
    /// `event_tail` bindings.
    pub event_ring: &'a crate::gpu::EventRing,
    /// Packed AbilityRegistry uploaded to GPU — `ability_registry_<col>`
    /// bindings.
    pub registry: &'a crate::ability::registry_gpu::PackedAbilityRegistryGpu,
    /// **Phase C** — optional GPU-resident voxel grid mirror. Resolves
    /// the `voxel_grid` binding name. `None` for runtimes that don't
    /// use voxel terrain (most fixtures today); the compiler-emitted
    /// `from_context` / `from_context_with_extras` bodies `.expect(...)`
    /// unwrap when a kernel actually binds `voxel_grid`, so leaving
    /// this `None` is safe for fixtures that don't lower terrain
    /// queries to GPU. Future fixtures opt in by passing
    /// `Some(mirror.buffer())`.
    pub voxel_grid: Option<&'a wgpu::Buffer>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_round_trips_standard_columns() {
        for col in AgentBuffers::STANDARD_COLUMNS {
            assert!(AgentBuffers::is_standard_column(col), "{col}");
        }
        assert!(!AgentBuffers::is_standard_column("creature_type"));
        assert!(!AgentBuffers::is_standard_column("disguise_fake_type"));
    }
}
