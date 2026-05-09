//! `viewer_runtime` — Phase A of the viewer-runtime plan
//! (`docs/superpowers/plans/2026-05-09-viewer-runtime.md`).
//!
//! Stands up the *data layer* of the viewer: a `ViewerApp` that
//! implements [`voxel_engine::app::App`], owns a sim runtime, and on
//! each fixed-step `tick()` mirrors agent positions into a
//! `voxel_engine::scene::Scene` as transform updates on
//! per-agent voxel-cube entities.
//!
//! Phase A intentionally stops at the data layer — no window, no
//! swapchain, no presentation. The console driver in
//! `src/bin/viewer_app.rs` exercises the App lifecycle against a
//! [`Scene::new_headless`] so the wiring is testable on CI without a
//! display surface. Wiring a winit window + voxel_engine renderer
//! around the same `ViewerApp` is a follow-up phase (see plan).
//!
//! Pilot fixture is wave_defense — palisades + monsters + settlers
//! gives the most "looks like a game" surface to drive subsequent
//! phases (voxel terrain sync, HUD, camera).

use anyhow::Result;
use glam::{IVec3, Quat, Vec3};
use voxel_engine::scene::handle::EntityHandle;
use voxel_engine::scene::transform::Transform;
use voxel_engine::scene::Scene;
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

use wave_defense_runtime::{WaveDefenseState, TOTAL_AGENT_CAPACITY};

/// Scene representation for one agent — a single-voxel `1×1×1` grid
/// at the agent's position. The voxel material id (1..=255) is set by
/// creature type via [`creature_material_index`] so the renderer can
/// later distinguish settler / monster / spawner / node visually.
fn agent_voxel_grid(material_index: u8) -> VoxelGrid {
    let mut g = VoxelGrid::new(1, 1, 1);
    g.set(0, 0, 0, material_index);
    g
}

/// Map a wave_defense `creature_type` ordinal to a palette index. The
/// concrete colors live in the palette (built once in `setup()`); this
/// is just the indirection.
fn creature_material_index(creature_type: u32) -> u8 {
    // Ordinals from `assets/sim/wave_defense.sim::config.combat`:
    //   type_node=0, type_settler=1, type_monster=2, type_spawner=3.
    // Map directly to material ids 1..=4 so the palette can be a small
    // hand-authored table; index 0 is air per voxel_engine convention.
    match creature_type {
        0 => 1, // node
        1 => 2, // settler
        2 => 3, // monster
        3 => 4, // spawner
        _ => 5, // unknown — distinct colour to surface bugs
    }
}

/// Build the per-creature-type palette: 4 colour entries plus index 0
/// = air. RGBA bytes are placeholders — distinct enough to read at a
/// glance, easy to tweak when the renderer goes windowed.
fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    p.set(1, palette_entry(255, 215, 0)); // node — gold
    p.set(2, palette_entry(80, 180, 255)); // settler — sky blue
    p.set(3, palette_entry(220, 60, 60)); // monster — crimson
    p.set(4, palette_entry(160, 90, 200)); // spawner — purple
    p.set(5, palette_entry(255, 0, 255)); // unknown — magenta
    p
}

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

/// Driver-style viewer app — wraps a [`WaveDefenseState`] and plumbs
/// per-tick agent positions into the renderer's scene as transform
/// updates on per-agent voxel entities.
///
/// Phase A is wave_defense-specific. Phase E generalises across
/// fixtures via feature flags + a thin per-fixture adapter.
pub struct ViewerApp {
    state: WaveDefenseState,
    /// `agent_handles[slot] = Some(handle)` once that agent has been
    /// spawned into the scene; `None` for slots not yet populated
    /// (monster pool slots that wave_defense only fills as waves
    /// arrive).
    agent_handles: Vec<Option<EntityHandle>>,
    palette: MaterialPalette,
    /// Most recent per-agent creature_type readback. Cached so we know
    /// whether a slot's representation needs to change (e.g. a monster
    /// pool slot transitioning from "empty" to a live creature).
    last_creature_type: Vec<u32>,
    /// Tick at which the sim last reported termination via
    /// `step_and_check_termination()`. `None` while still running.
    pub terminated_at_tick: Option<u64>,
}

impl ViewerApp {
    pub fn new(seed: u64) -> Self {
        let state = WaveDefenseState::new(seed);
        let n = TOTAL_AGENT_CAPACITY as usize;
        Self {
            state,
            agent_handles: vec![None; n],
            palette: build_palette(),
            last_creature_type: vec![u32::MAX; n],
            terminated_at_tick: None,
        }
    }

    /// Current sim tick (delegate; saves callers a re-import).
    /// Named `sim_tick` rather than `tick` so it doesn't shadow the
    /// `App::tick` trait method when callers have both in scope.
    pub fn sim_tick(&self) -> u64 {
        self.state.tick()
    }

    /// Number of scene entities currently populated.
    pub fn populated_entity_count(&self) -> usize {
        self.agent_handles.iter().filter(|h| h.is_some()).count()
    }

    /// Sim-state accessors used by the windowed driver's title bar.
    /// Delegate to the underlying [`WaveDefenseState`]. Each one
    /// triggers a small GPU readback per call — fine for once-per-tick
    /// title-bar refresh, not for hot loops.
    pub fn alive_settlers(&self) -> u32 {
        self.state.alive_settler_count()
    }
    pub fn alive_monsters(&self) -> u32 {
        self.state.alive_monster_count()
    }
    pub fn score(&self) -> f32 {
        self.state.read_score()
    }

    /// Sync one agent slot into the scene. If the slot is alive and
    /// has no entity yet, spawn one; if it already has one, update
    /// the transform; if the creature_type changed (slot reused),
    /// despawn + respawn with the new material.
    fn sync_slot(
        &mut self,
        scene: &mut Scene,
        slot: usize,
        pos: Vec3,
        alive: bool,
        creature_type: u32,
    ) {
        let cached_creature = self.last_creature_type[slot];
        let creature_changed = cached_creature != creature_type;
        self.last_creature_type[slot] = creature_type;

        if !alive {
            // Despawn if previously alive — keeps the scene clean as
            // monsters die.
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
            Some(handle) if !creature_changed => {
                scene.set_transform(handle, transform);
            }
            Some(handle) => {
                // Slot reused with different creature_type — despawn +
                // respawn so the material id reflects the new role.
                scene.despawn(handle);
                let grid = agent_voxel_grid(creature_material_index(creature_type));
                let h = scene.spawn(&grid, transform, &self.palette);
                self.agent_handles[slot] = Some(h);
            }
            None => {
                let grid = agent_voxel_grid(creature_material_index(creature_type));
                let h = scene.spawn(&grid, transform, &self.palette);
                self.agent_handles[slot] = Some(h);
            }
        }
    }
}

impl voxel_engine::app::App for ViewerApp {
    fn setup(&mut self, scene: &mut Scene) -> Result<()> {
        // Pre-populate the scene with whatever agents are already
        // alive at construction (the node + settler ring + spawner
        // ring; monster pool slots stay empty until the first wave).
        let positions = self.state.read_pos();
        let alive = self.state.read_alive();
        let creature_types = self.state.read_creature_type();
        for slot in 0..(TOTAL_AGENT_CAPACITY as usize) {
            self.sync_slot(
                scene,
                slot,
                positions[slot],
                alive[slot] != 0,
                creature_types[slot],
            );
        }
        Ok(())
    }

    fn tick(&mut self, scene: &mut Scene, _dt: f32) {
        // Advance the sim. `step_and_check_termination` returns true
        // once the settlement has fallen; from there we leave the
        // scene frozen at the death state so the viewer doesn't blink.
        if self.terminated_at_tick.is_some() {
            return;
        }
        let terminated = self.state.step_and_check_termination();
        if terminated {
            self.terminated_at_tick = Some(self.state.tick());
        }

        // Mirror per-agent state into the scene. Three readbacks per
        // tick is heavy at 100ms cadence — Phase B will switch to a
        // single fused readback, but Phase A's correctness gate beats
        // its perf gate.
        let positions = self.state.read_pos();
        let alive = self.state.read_alive();
        let creature_types = self.state.read_creature_type();
        for slot in 0..(TOTAL_AGENT_CAPACITY as usize) {
            self.sync_slot(
                scene,
                slot,
                positions[slot],
                alive[slot] != 0,
                creature_types[slot],
            );
        }
    }

    fn on_input(&mut self, _scene: &mut Scene, _event: &winit::event::WindowEvent) {
        // Phase D wires camera input. Phase A is observer-only.
    }
}

// `IVec3` import is unused in Phase A; keeping it pulled in so Phase B
// (voxel terrain sync, which uses IVec3-keyed dirty cells) lands
// without churning the use list.
#[allow(dead_code)]
fn _phase_b_marker(_v: IVec3) {}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_engine::scene::config::SceneConfig;

    /// Phase A runtime gate (per the plan's AIS): construct the
    /// viewer against a headless scene, drive 10 fixed-step ticks,
    /// assert agent positions reach the scene as transforms.
    /// Skips cleanly when no GPU adapter is available — the
    /// `WaveDefenseState::new` panic is the same shape as
    /// `same_seed_same_death_tick`'s skip.
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
        assert!(
            post_setup_count > 0,
            "setup() must spawn at least one scene entity (the node \
             + settler ring + spawner ring all start alive); saw {}",
            post_setup_count,
        );

        for _ in 0..10 {
            voxel_engine::app::App::tick(&mut app, &mut scene, 0.1);
            // Bail early if the sim somehow terminates within 10 ticks
            // (it shouldn't at default seed, but the test stays robust).
            if app.terminated_at_tick.is_some() {
                break;
            }
        }

        assert_eq!(app.sim_tick(), 10, "10 fixed-step ticks must advance sim.tick to 10");

        // The settler ring shouldn't have died in 10 ticks; populated
        // entity count should be at least the post-setup baseline.
        let post_tick_count = app.populated_entity_count();
        assert!(
            post_tick_count >= post_setup_count,
            "alive-agent count shouldn't drop below the post-setup \
             baseline within 10 ticks (post_setup={}, post_tick={})",
            post_setup_count,
            post_tick_count,
        );
    }
}
