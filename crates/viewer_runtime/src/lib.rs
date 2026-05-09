//! `viewer_runtime` — windowed viewer over a deterministic sim.
//!
//! Pilot fixture: `objective_capture_10v10_runtime` (10v10 territory
//! control, single objective at world origin). 20 total agents; Red
//! team painted red, Blue team painted blue, plus a yellow marker at
//! the objective.
//!
//! Phase A established the data layer (App impl + scene-mirror
//! against `Scene::new_headless`).
//! Phase A.5 stood up windowed presentation via voxel_engine's
//! Vulkan renderer.
//! Phase B brought agents on-screen as voxel cells via
//! [`VoxelBridge`].
//! Phase C added the egui HUD overlay.
//! This swap (2026-05-09) replaced the prior wave_defense pilot,
//! which under the current tuning was too combat-asymmetric to
//! produce visually interesting behaviour. Multi-fixture support
//! (keep both, pick at runtime) is the Phase E slice in the plan
//! at `docs/superpowers/plans/2026-05-09-viewer-runtime.md`.

use anyhow::Result;
use glam::{IVec3, Quat, Vec3};
use voxel_engine::scene::handle::EntityHandle;
use voxel_engine::scene::transform::Transform;
use voxel_engine::scene::Scene;
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

use engine::CompiledSim;
use objective_capture_10v10_runtime::{
    ObjectiveCapture10v10State, ObjectiveState, OBJECTIVE_POS, TEAM_SIZE,
};

pub mod voxel_bridge;
pub use voxel_bridge::VoxelBridge;

/// One scene entity per agent — a single-voxel `1×1×1` grid at the
/// agent's position. Material id (1..=255) selects a palette entry.
fn agent_voxel_grid(material_index: u8) -> VoxelGrid {
    let mut g = VoxelGrid::new(1, 1, 1);
    g.set(0, 0, 0, material_index);
    g
}

/// Map a fixture-specific concept (team / role) to a palette index.
/// Material id 0 is air per voxel_engine convention.
///
/// objective_capture_10v10 has two teams + one stationary objective
/// marker. Team ordinal `0 = Red`, `1 = Blue`; the objective gets
/// its own slot so the player can see "the thing being contested"
/// without it sharing a colour with either team.
fn team_material_index(team: u8) -> u8 {
    match team {
        0 => 1, // Red team
        1 => 2, // Blue team
        _ => 5, // unknown / surface bugs
    }
}

const OBJECTIVE_MATERIAL: u8 = 3;
const UNKNOWN_MATERIAL: u8 = 5;

fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    p.set(1, palette_entry(220, 60, 60)); // Red team
    p.set(2, palette_entry(60, 120, 220)); // Blue team
    p.set(3, palette_entry(255, 215, 0)); // objective — gold
    p.set(GROUND_MATERIAL, palette_entry(90, 80, 70)); // muted brown — ground
    p.set(UNKNOWN_MATERIAL, palette_entry(255, 0, 255)); // magenta — surface bugs
    p
}

/// Material id for the ground plane (painted once at construction).
pub const GROUND_MATERIAL: u8 = 4;

fn palette_entry(r: u8, g: u8, b: u8) -> voxel_engine::voxel::material::PaletteEntry {
    voxel_engine::voxel::material::PaletteEntry {
        r,
        g,
        b,
        roughness: 200,
        emissive: 0,
        material_type: voxel_engine::voxel::material::MaterialType::Stone,
    }
}

/// Wraps the sim runtime + maintains snapshots of per-agent state
/// the voxel bridge consumes each tick. Implements
/// [`voxel_engine::app::App`] so the windowed runner drives it via
/// `setup() / tick() / on_input()`.
pub struct ViewerApp {
    state: ObjectiveCapture10v10State,
    /// agent_handles[slot] = Some(handle) once that agent has been
    /// spawned into the scene. objective_capture has TEAM_SIZE×2
    /// agents, all populated at construction.
    agent_handles: Vec<Option<EntityHandle>>,
    palette: MaterialPalette,
    /// Cached per-agent positions / alive / material so the bridge
    /// can read without re-issuing host work each frame.
    last_positions: Vec<Vec3>,
    last_alive: Vec<u32>,
    last_material: Vec<u8>,
    /// Tick at which the sim reported a winner; from then on the
    /// scene freezes at the final state.
    pub terminated_at_tick: Option<u64>,
    /// Most recent objective-state readback (control percentages,
    /// scores). Surfaced via accessors for the HUD.
    last_objective: ObjectiveState,
}

impl ViewerApp {
    pub fn new(seed: u64) -> Self {
        let state = ObjectiveCapture10v10State::new(seed);
        let n = state.agent_count() as usize;
        Self {
            state,
            agent_handles: vec![None; n],
            palette: build_palette(),
            last_positions: vec![Vec3::ZERO; n],
            last_alive: vec![0; n],
            last_material: vec![UNKNOWN_MATERIAL; n],
            terminated_at_tick: None,
            last_objective: ObjectiveState {
                red_alive: 0,
                blue_alive: 0,
                red_in_zone: 0,
                blue_in_zone: 0,
                red_score: 0,
                blue_score: 0,
            },
        }
    }

    pub fn sim_tick(&self) -> u64 {
        self.state.tick()
    }

    pub fn populated_entity_count(&self) -> usize {
        self.agent_handles.iter().filter(|h| h.is_some()).count()
    }

    pub fn red_score(&self) -> u32 {
        self.state.red_score()
    }
    pub fn blue_score(&self) -> u32 {
        self.state.blue_score()
    }
    pub fn red_alive(&self) -> u32 {
        self.last_alive[..(TEAM_SIZE as usize)]
            .iter()
            .filter(|&&a| a != 0)
            .count() as u32
    }
    pub fn blue_alive(&self) -> u32 {
        self.last_alive[(TEAM_SIZE as usize)..]
            .iter()
            .filter(|&&a| a != 0)
            .count() as u32
    }
    pub fn winner(&self) -> Option<u8> {
        self.state.winner()
    }
    pub fn objective_state(&self) -> &ObjectiveState {
        &self.last_objective
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.last_positions
    }
    pub fn alive(&self) -> &[u32] {
        &self.last_alive
    }
    pub fn materials(&self) -> &[u8] {
        &self.last_material
    }
    pub fn palette(&self) -> &MaterialPalette {
        &self.palette
    }

    /// Backwards-compatible accessor — voxel_bridge calls
    /// `app.material_for(creature_type)` from the wave_defense
    /// era. With the new fixture, `materials()` is the canonical
    /// path; this stays as a courtesy passthrough that ignores
    /// its argument and returns the cached per-slot value if
    /// the slot index is encoded as the argument's low bits.
    pub fn material_for(&self, ordinal_or_slot: u32) -> u8 {
        // Treat the arg as a slot index when it fits; falls back
        // to UNKNOWN_MATERIAL otherwise. Bridge has been updated
        // to call `materials()[slot]` directly so this path
        // shouldn't fire under normal operation.
        self.last_material
            .get(ordinal_or_slot as usize)
            .copied()
            .unwrap_or(UNKNOWN_MATERIAL)
    }

    fn refresh_snapshot(&mut self) {
        // Host-side state — no GPU readback required.
        // **Axis swap**: sim is Z-up (movement on XY plane, Z is
        // vertical), voxel_engine is Y-up (sun_dir.y dominant,
        // hemisphere ambient keyed on normal.y). Without the swap
        // the renderer treats the sim's ground plane as a wall.
        // Swap (sim.x, sim.y, sim.z) → (voxel.x, voxel.z, voxel.y)
        // so vertical lines up across both worlds.
        let host_pos = self.state.positions();
        let host_alive = self.state.host_alive();
        let host_teams = self.state.teams();
        for slot in 0..self.last_positions.len() {
            let s = host_pos[slot];
            self.last_positions[slot] = Vec3::new(s.x, s.z, s.y);
            self.last_alive[slot] = if host_alive[slot] { 1 } else { 0 };
            self.last_material[slot] = team_material_index(host_teams[slot]);
        }
        self.last_objective = self.state.read_objective_state();
    }

    fn sync_slot(
        &mut self,
        scene: &mut Scene,
        slot: usize,
        pos: Vec3,
        alive: bool,
        material: u8,
    ) {
        if !alive {
            if let Some(handle) = self.agent_handles[slot].take() {
                scene.despawn(handle);
            }
            return;
        }
        let transform = Transform {
            position: pos,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        match self.agent_handles[slot] {
            Some(handle) => {
                scene.set_transform(handle, transform);
            }
            None => {
                let grid = agent_voxel_grid(material);
                let h = scene.spawn(&grid, transform, &self.palette);
                self.agent_handles[slot] = Some(h);
            }
        }
    }
}

impl voxel_engine::app::App for ViewerApp {
    fn setup(&mut self, scene: &mut Scene) -> Result<()> {
        self.refresh_snapshot();
        for slot in 0..self.last_positions.len() {
            self.sync_slot(
                scene,
                slot,
                self.last_positions[slot],
                self.last_alive[slot] != 0,
                self.last_material[slot],
            );
        }
        Ok(())
    }

    fn tick(&mut self, scene: &mut Scene, _dt: f32) {
        if self.terminated_at_tick.is_some() {
            return;
        }
        self.state.step();
        if self.state.winner().is_some() && self.terminated_at_tick.is_none() {
            self.terminated_at_tick = Some(self.state.tick());
        }
        self.refresh_snapshot();
        for slot in 0..self.last_positions.len() {
            self.sync_slot(
                scene,
                slot,
                self.last_positions[slot],
                self.last_alive[slot] != 0,
                self.last_material[slot],
            );
        }
    }

    fn on_input(&mut self, _scene: &mut Scene, _event: &winit::event::WindowEvent) {
        // Camera controls are a separate phase. ViewerApp is
        // observer-only.
    }
}

/// Constant exposed so the bridge / window driver can place a
/// stationary marker at the objective without needing to import
/// the runtime crate themselves. Same Y/Z axis swap as agent
/// positions so the marker lands on the rendered ground plane.
pub fn objective_world_position() -> Vec3 {
    Vec3::new(OBJECTIVE_POS.x, OBJECTIVE_POS.z, OBJECTIVE_POS.y)
}

/// Material id used for the objective marker — distinct from any
/// team colour so it reads as "the thing being contested".
pub fn objective_material() -> u8 {
    OBJECTIVE_MATERIAL
}

#[allow(dead_code)]
fn _phase_b_marker(_v: IVec3) {}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::scene::config::SceneConfig;

    #[test]
    fn viewer_construction_succeeds_or_skips() {
        let init = std::panic::catch_unwind(|| ViewerApp::new(0xCAFE_F00D));
        let mut app = match init {
            Ok(a) => a,
            Err(_) => {
                eprintln!(
                    "[viewer_construction_succeeds_or_skips] skipping: GPU init failed"
                );
                return;
            }
        };

        let mut scene = Scene::new_headless(SceneConfig::default());
        voxel_engine::app::App::setup(&mut app, &mut scene)
            .expect("setup() must succeed against a headless scene");

        let post_setup_count = app.populated_entity_count();
        assert_eq!(
            post_setup_count,
            (TEAM_SIZE * 2) as usize,
            "objective_capture spawns 2 × TEAM_SIZE agents at construction; \
             saw {} populated",
            post_setup_count,
        );

        for _ in 0..10 {
            voxel_engine::app::App::tick(&mut app, &mut scene, 0.1);
            if app.terminated_at_tick.is_some() {
                break;
            }
        }

        assert_eq!(
            app.sim_tick(),
            10,
            "10 fixed-step ticks must advance sim.tick to 10",
        );
    }
}
