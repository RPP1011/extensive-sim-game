//! `engine_play` — the generic, descriptor-driven player.
//!
//! One binary plays any compiled `.sim` runtime: a `RenderDescriptor` says how
//! agents map to voxel colors + VFX, a `ControlsDescriptor` maps keys to
//! `@runtime` input writes, and an `engine_ui::UiModel` declares the HUD/menu/
//! death overlay. All three are consumed through the `engine_play_api`
//! [`PlayableRuntime`](engine_play_api::PlayableRuntime) trait, so this crate
//! has no dependency on the compiler emitting anything — it is tested against a
//! [`MockRuntime`](mock::MockRuntime) plus `engine_play_api`'s sample fixtures.
//!
//! Modules:
//! - [`input`] — the pure `ControlsMapper` (keys + descriptor → input writes).
//! - [`bridge`] — the `EngineBridge`, a generalized port of
//!   `viewer_runtime::vs::VsBridge` driven by a `RenderDescriptor` + per-frame
//!   `AgentView` snapshot.
//! - [`player`] — the per-frame `update()` (separable from the winit/GPU shell)
//!   plus the windowed `Player` loop.
//! - [`mock`] — a `MockRuntime` for headless tests.

pub mod bridge;
pub mod input;
pub mod mock;
pub mod player;

pub use bridge::{EngineBridge, PaintGrid, Painted};
pub use input::ControlsMapper;
pub use player::{update, HostState, Player, PlayerConfig, UpdateOutput};
