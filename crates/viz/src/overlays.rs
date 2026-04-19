//! Event overlays. Per-frame: ingest new events → record Overlay → prune
//! expired → paint all live overlays into the grid.

use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;

use crate::grid_paint::{paint_line, paint_ring};
use crate::palette::{PAL_ANNOUNCE, PAL_ATTACK, PAL_DEATH};

#[derive(Debug, Clone, Copy)]
pub enum OverlayKind {
    AttackLine  { from: Vec3, to: Vec3 },
    DeathMarker { at: Vec3 },
    AnnounceRing { speaker: Vec3, born_tick: u32, max_radius: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct Overlay {
    pub kind:            OverlayKind,
    pub born_tick:       u32,
    pub expires_at_tick: u32,
}

pub const ATTACK_LINE_TTL_TICKS:   u32 = 5;
pub const ANNOUNCE_RING_TTL_TICKS: u32 = 3;
/// Matches `engine::step::MAX_ANNOUNCE_RADIUS`.
pub const DEFAULT_ANNOUNCE_RADIUS: f32 = 80.0;

pub struct OverlayTracker {
    overlays: Vec<Overlay>,
    /// Highest tick we've already converted to overlays; future events
    /// must have `tick > last_scanned_tick` to be processed.
    last_scanned_tick: u32,
}

impl Default for OverlayTracker {
    fn default() -> Self { Self::new() }
}

impl OverlayTracker {
    pub fn new() -> Self {
        Self { overlays: Vec::with_capacity(64), last_scanned_tick: 0 }
    }

    /// Walk `events.iter()` (non-destructive — chronicle needs to re-read
    /// them) and record overlays for AgentAttacked / AgentDied /
    /// AnnounceEmitted. Pulls attacker/target/speaker positions from
    /// `state` because events don't carry positions.
    pub fn ingest_with_state(
        &mut self,
        events: &engine::event::EventRing,
        state:  &engine::state::SimState,
    ) {
        use engine::event::Event;
        let current_tick = state.tick;
        for e in events.iter() {
            if e.tick() <= self.last_scanned_tick { continue; }
            match *e {
                Event::AgentAttacked { attacker, target, tick, .. } => {
                    let from = state.agent_pos(attacker).unwrap_or(Vec3::ZERO);
                    let to   = state.agent_pos(target).unwrap_or(Vec3::ZERO);
                    self.overlays.push(Overlay {
                        kind: OverlayKind::AttackLine { from, to },
                        born_tick: tick,
                        expires_at_tick: tick.saturating_add(ATTACK_LINE_TTL_TICKS),
                    });
                }
                Event::AgentDied { agent_id, tick } => {
                    let at = state.agent_pos(agent_id).unwrap_or(Vec3::ZERO);
                    self.overlays.push(Overlay {
                        kind: OverlayKind::DeathMarker { at },
                        born_tick: tick,
                        expires_at_tick: u32::MAX,
                    });
                }
                Event::AnnounceEmitted { speaker, tick, .. } => {
                    let pos = state.agent_pos(speaker).unwrap_or(Vec3::ZERO);
                    self.overlays.push(Overlay {
                        kind: OverlayKind::AnnounceRing {
                            speaker: pos, born_tick: tick,
                            max_radius: DEFAULT_ANNOUNCE_RADIUS,
                        },
                        born_tick: tick,
                        expires_at_tick: tick.saturating_add(ANNOUNCE_RING_TTL_TICKS),
                    });
                }
                _ => {}
            }
        }
        self.last_scanned_tick = current_tick;
    }

    pub fn prune(&mut self, current_tick: u32) {
        self.overlays.retain(|o| current_tick <= o.expires_at_tick);
    }

    pub fn paint_into(&self, grid: &mut VoxelGrid, current_tick: u32) {
        for o in &self.overlays {
            match o.kind {
                OverlayKind::AttackLine { from, to } => {
                    paint_line(grid, from, to, PAL_ATTACK);
                }
                OverlayKind::DeathMarker { at } => {
                    // single-voxel via the degenerate paint_line path
                    paint_line(grid, at, at, PAL_DEATH);
                }
                OverlayKind::AnnounceRing { speaker, born_tick, max_radius } => {
                    let ttl = ANNOUNCE_RING_TTL_TICKS.max(1) as f32;
                    let age = current_tick.saturating_sub(born_tick) as f32;
                    let frac = (age / ttl).clamp(0.0, 1.0);
                    paint_ring(grid, speaker, max_radius * frac, PAL_ANNOUNCE);
                }
            }
        }
    }

    pub fn len(&self) -> usize { self.overlays.len() }
    pub fn is_empty(&self) -> bool { self.overlays.is_empty() }
}
