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
/// Fallback announce-ring radius when the viz layer has no live `SimState`
/// to read `config.communication.max_announce_radius` from. Matches the
/// DSL-shipped default in `assets/sim/config.sim`; TOML tuning overrides
/// the sim-side radius while the viz keeps painting at this fixed fallback
/// — intentionally viz-primitive (painting aid, not a balance knob).
pub const DEFAULT_ANNOUNCE_RADIUS: f32 = 80.0;

pub struct OverlayTracker {
    overlays: Vec<Overlay>,
    /// Monotonic cursor: the next undiscovered push-index into the event
    /// ring. Advancing this instead of a tick cursor lets us capture
    /// multiple same-tick events (e.g. both the AgentAttacked that
    /// brings HP to 0 AND the AgentDied that follows it) without the
    /// off-by-one hazards of `tick <=/< last_scanned_tick`.
    next_event_idx: usize,
}

impl Default for OverlayTracker {
    fn default() -> Self { Self::new() }
}

impl OverlayTracker {
    pub fn new() -> Self {
        Self { overlays: Vec::with_capacity(64), next_event_idx: 0 }
    }

    /// Walk `events` (non-destructive — chronicle needs to re-read them)
    /// and record overlays for AgentAttacked / AgentDied / AnnounceEmitted.
    /// Pulls attacker/target/speaker positions from `state` because events
    /// don't carry positions.
    pub fn ingest_with_state(
        &mut self,
        events: &engine::event::EventRing,
        state:  &engine::state::SimState,
    ) {
        use engine::event::Event;
        let total = events.total_pushed();
        while self.next_event_idx < total {
            let Some(e) = events.get_pushed(self.next_event_idx) else {
                // evicted before we could see it — skip ahead to the
                // oldest still-resident entry.
                let first_resident = total.saturating_sub(events.len());
                self.next_event_idx = first_resident;
                continue;
            };
            match e {
                Event::AgentAttacked { actor, target, tick, .. } => {
                    let from = state.agent_pos(actor).unwrap_or(Vec3::ZERO);
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
            self.next_event_idx += 1;
        }
        let _ = state; // current_tick is no longer used; state kept on signature for API stability.
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

    /// Paint attack lines and announce rings, but NOT death markers. The
    /// caller is responsible for painting death markers as part of the
    /// agent-stacking pass so they get stacked with any alive agents at
    /// the same (x, z) voxel cell. See `state::render` for the stacking
    /// logic and `death_positions` for the companion iterator.
    pub fn paint_non_death(&self, grid: &mut VoxelGrid, current_tick: u32) {
        for o in &self.overlays {
            match o.kind {
                OverlayKind::AttackLine { from, to } => {
                    paint_line(grid, from, to, PAL_ATTACK);
                }
                OverlayKind::DeathMarker { .. } => { /* handled by stacking pass */ }
                OverlayKind::AnnounceRing { speaker, born_tick, max_radius } => {
                    let ttl = ANNOUNCE_RING_TTL_TICKS.max(1) as f32;
                    let age = current_tick.saturating_sub(born_tick) as f32;
                    let frac = (age / ttl).clamp(0.0, 1.0);
                    paint_ring(grid, speaker, max_radius * frac, PAL_ANNOUNCE);
                }
            }
        }
    }

    /// Iterator over positions of all currently-live death marker overlays.
    /// Used by the stacking pass so death markers end up above any alive
    /// agents sharing their (x, z) cell rather than getting hidden under
    /// them (and vice-versa).
    pub fn death_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.overlays.iter().filter_map(|o| match o.kind {
            OverlayKind::DeathMarker { at } => Some(at),
            _ => None,
        })
    }

    pub fn len(&self) -> usize { self.overlays.len() }
    pub fn is_empty(&self) -> bool { self.overlays.is_empty() }
}
