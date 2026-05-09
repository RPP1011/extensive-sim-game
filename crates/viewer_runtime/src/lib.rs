//! `viewer_runtime` — windowed viewer over a deterministic sim.
//!
//! Pilot fixture: `boss_fight_runtime`. 1 Boss (slot 0,
//! creature_type=0) + 5 Heroes (slots 1..=5, creature_type=1)
//! trading abilities (BossStrike / BossSelfHeal / HeroAttack /
//! HeroStun / HeroHeal). The sim has no per-agent movement —
//! viewer assigns a fixed semicircle layout. Visual interest comes
//! from per-tick HP swings + stun status, surfaced via cube colour
//! (tinted by HP fraction) and an egui panel with per-unit HP bars.
//!
//! Earlier pilots: wave_defense (too combat-asymmetric to look
//! interesting), objective_capture_10v10 (10v10 capture-the-point —
//! still in the workspace, swap path needs Phase E multi-fixture
//! support per `docs/superpowers/plans/2026-05-09-viewer-runtime.md`).

use anyhow::Result;
use glam::{IVec3, Quat, Vec3};
use voxel_engine::scene::handle::EntityHandle;
use voxel_engine::scene::transform::Transform;
use voxel_engine::scene::Scene;
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

use boss_fight_runtime::BossFightState;
use engine::CompiledSim;

pub mod voxel_bridge;
pub use voxel_bridge::VoxelBridge;

/// Per-fixture agent count. Boss + 5 Heroes.
const AGENT_COUNT: u32 = 6;
const BOSS_SLOT: usize = 0;

/// Hero positions in sim coords (Z-up), arranged in a semicircle
/// facing the boss at origin. Boss is at sim (0,0,0); heroes are at
/// radius 12 along the +X half-plane, evenly distributed in the
/// Y axis (their "side-to-side" axis when viewed top-down).
const HERO_POSITIONS: [Vec3; 5] = [
    Vec3::new(12.0, -8.0, 0.0),
    Vec3::new(15.0, -4.0, 0.0),
    Vec3::new(16.0, 0.0, 0.0),
    Vec3::new(15.0, 4.0, 0.0),
    Vec3::new(12.0, 8.0, 0.0),
];
const BOSS_POSITION: Vec3 = Vec3::new(-6.0, 0.0, 0.0);

/// One scene entity per agent — a single-voxel `1×1×1` grid.
fn agent_voxel_grid(material_index: u8) -> VoxelGrid {
    let mut g = VoxelGrid::new(1, 1, 1);
    g.set(0, 0, 0, material_index);
    g
}

/// Map (creature_type, hp_fraction) → palette index. We populate
/// the palette with multiple shades of each role's base colour,
/// indexed by HP bucket. Bucket 0 = nearly dead (dim), bucket
/// `HP_BUCKETS-1` = full health (vivid).
fn material_for(creature_type: u32, hp_fraction: f32) -> u8 {
    let role_base = match creature_type {
        0 => 1,                    // Boss base
        1 => 1 + HP_BUCKETS,       // Hero base
        _ => 1 + 2 * HP_BUCKETS,   // unknown
    };
    let frac = hp_fraction.clamp(0.0, 1.0);
    let bucket = ((frac * (HP_BUCKETS as f32 - 1.0)).round() as u8).min(HP_BUCKETS as u8 - 1);
    role_base + bucket
}

const HP_BUCKETS: u8 = 4;
const STUN_MATERIAL: u8 = 1 + 3 * HP_BUCKETS; // single shade for stunned overlay
const GROUND_MATERIAL: u8 = STUN_MATERIAL + 1;
/// Effect flash materials — used for the ability splats that
/// hover above an agent for `EFFECT_DURATION_TICKS` after the
/// matching event fires.
pub const EFFECT_DAMAGE_MATERIAL: u8 = STUN_MATERIAL + 2;
pub const EFFECT_HEAL_MATERIAL: u8 = STUN_MATERIAL + 3;
pub const EFFECT_STUN_MATERIAL: u8 = STUN_MATERIAL + 4;
const UNKNOWN_MATERIAL: u8 = STUN_MATERIAL + 5;

/// How many fixed-step ticks an ability flash stays visible.
/// 3 ticks = 300ms — long enough to register, short enough to
/// not mask the underlying agent state.
pub const EFFECT_DURATION_TICKS: u64 = 3;

/// Source of an effect flash — drives the colour material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    Damage,
    Heal,
    Stun,
}

impl EffectKind {
    pub fn material(self) -> u8 {
        match self {
            EffectKind::Damage => EFFECT_DAMAGE_MATERIAL,
            EffectKind::Heal => EFFECT_HEAL_MATERIAL,
            EffectKind::Stun => EFFECT_STUN_MATERIAL,
        }
    }
}

/// A short-lived flash placed at an agent's position.
#[derive(Debug, Clone, Copy)]
pub struct EffectMarker {
    pub slot: usize,
    pub kind: EffectKind,
    pub expires_at_tick: u64,
}

fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    // Boss role — crimson red, 4 brightness buckets (dying → full).
    let boss_base = (140, 30, 30);
    let hero_base = (60, 130, 220);
    let unk_base = (255, 0, 255);
    fill_role_buckets(&mut p, 1, boss_base);
    fill_role_buckets(&mut p, 1 + HP_BUCKETS, hero_base);
    fill_role_buckets(&mut p, 1 + 2 * HP_BUCKETS, unk_base);
    p.set(STUN_MATERIAL, palette_entry(255, 215, 80)); // stun glow — yellow
    p.set(GROUND_MATERIAL, palette_entry(90, 80, 70));
    p.set(EFFECT_DAMAGE_MATERIAL, palette_entry(255, 80, 80)); // damage flash — bright red
    p.set(EFFECT_HEAL_MATERIAL, palette_entry(80, 255, 120));  // heal flash — bright green
    p.set(EFFECT_STUN_MATERIAL, palette_entry(255, 240, 60));  // stun pulse — bright yellow
    p.set(UNKNOWN_MATERIAL, palette_entry(255, 0, 255));
    p
}

fn fill_role_buckets(p: &mut MaterialPalette, base_id: u8, (r, g, b): (u8, u8, u8)) {
    for i in 0..HP_BUCKETS {
        // Bucket 0 = darkest (dying), bucket HP_BUCKETS-1 = full.
        let t = (i as f32 + 0.6) / (HP_BUCKETS as f32);
        let scale = |v: u8| ((v as f32 * t).round() as u8).min(255);
        p.set(base_id + i, palette_entry(scale(r), scale(g), scale(b)));
    }
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

/// Wraps the sim runtime + maintains snapshots of per-agent state
/// the voxel bridge consumes each tick.
pub struct ViewerApp {
    state: BossFightState,
    agent_handles: Vec<Option<EntityHandle>>,
    palette: MaterialPalette,
    /// Position is fixed per-slot — boss + 5 heroes don't move
    /// in this fixture. Pre-computed once + recomputed only if
    /// agent_count changes (which it doesn't today).
    last_positions: Vec<Vec3>,
    last_alive: Vec<u32>,
    last_hp: Vec<f32>,
    last_max_hp: Vec<f32>,
    last_creature_type: Vec<u32>,
    last_stun_expires: Vec<u32>,
    last_material: Vec<u8>,
    /// Per-slot HP from the previous tick — diffed against the
    /// current readback to detect damage / heal events without
    /// any sim-side instrumentation.
    prev_hp: Vec<f32>,
    /// Per-slot stun expiry from the previous tick — used to
    /// detect a fresh stun (expiry rose past current tick).
    prev_stun_expires: Vec<u32>,
    /// Active effect flashes, painted by the bridge for
    /// `EFFECT_DURATION_TICKS` ticks then removed.
    effects: Vec<EffectMarker>,
    pub terminated_at_tick: Option<u64>,
}

impl ViewerApp {
    pub fn new(seed: u64) -> Self {
        let state = BossFightState::new(seed, AGENT_COUNT);
        let n = AGENT_COUNT as usize;
        let mut last_positions = vec![Vec3::ZERO; n];
        // Sim is Z-up; voxel-engine is Y-up. Apply the swap when
        // we cache the layout (BOSS_POSITION + HERO_POSITIONS are
        // in sim-coord convention). After this, `last_positions`
        // is in voxel-coord convention.
        last_positions[BOSS_SLOT] =
            Vec3::new(BOSS_POSITION.x, BOSS_POSITION.z, BOSS_POSITION.y);
        for (i, hpos) in HERO_POSITIONS.iter().enumerate() {
            last_positions[i + 1] = Vec3::new(hpos.x, hpos.z, hpos.y);
        }
        Self {
            state,
            agent_handles: vec![None; n],
            palette: build_palette(),
            last_positions,
            last_alive: vec![0; n],
            last_hp: vec![0.0; n],
            last_max_hp: vec![0.0; n],
            last_creature_type: vec![0; n],
            last_stun_expires: vec![0; n],
            last_material: vec![UNKNOWN_MATERIAL; n],
            prev_hp: vec![0.0; n],
            prev_stun_expires: vec![0; n],
            effects: Vec::with_capacity(32),
            terminated_at_tick: None,
        }
    }

    /// Active effect flashes (damage / heal / stun pulses).
    /// Painted by the bridge each frame; cleared automatically as
    /// `expires_at_tick` passes.
    pub fn effects(&self) -> &[EffectMarker] {
        &self.effects
    }

    pub fn sim_tick(&self) -> u64 {
        self.state.tick()
    }

    pub fn populated_entity_count(&self) -> usize {
        self.agent_handles.iter().filter(|h| h.is_some()).count()
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
    pub fn material_for(&self, slot_or_ordinal: u32) -> u8 {
        self.last_material
            .get(slot_or_ordinal as usize)
            .copied()
            .unwrap_or(UNKNOWN_MATERIAL)
    }

    /// HUD accessors.
    pub fn hp(&self) -> &[f32] {
        &self.last_hp
    }
    pub fn max_hp(&self) -> &[f32] {
        &self.last_max_hp
    }
    pub fn creature_types(&self) -> &[u32] {
        &self.last_creature_type
    }
    /// Per-agent stun expiry tick. `0` = not stunned. Compare to
    /// `sim_tick()` to decide if a slot is currently stunned.
    pub fn stun_expires(&self) -> &[u32] {
        &self.last_stun_expires
    }
    pub fn is_stunned(&self, slot: usize) -> bool {
        self.last_stun_expires
            .get(slot)
            .map(|&exp| exp as u64 > self.state.tick())
            .unwrap_or(false)
    }
    /// Sum HP of all alive heroes (slots 1..=5).
    pub fn party_total_hp(&self) -> f32 {
        (1..(AGENT_COUNT as usize))
            .filter(|&s| self.last_alive[s] != 0)
            .map(|s| self.last_hp[s])
            .sum()
    }
    pub fn party_max_total_hp(&self) -> f32 {
        (1..(AGENT_COUNT as usize)).map(|s| self.last_max_hp[s]).sum()
    }
    pub fn party_alive_count(&self) -> u32 {
        (1..(AGENT_COUNT as usize))
            .filter(|&s| self.last_alive[s] != 0)
            .count() as u32
    }
    pub fn boss_alive(&self) -> bool {
        self.last_alive[BOSS_SLOT] != 0
    }
    pub fn boss_hp(&self) -> (f32, f32) {
        (self.last_hp[BOSS_SLOT], self.last_max_hp[BOSS_SLOT])
    }

    fn refresh_snapshot(&mut self) {
        let hp = self.state.read_hp();
        let alive = self.state.read_alive();
        let creature_type = self.state.read_creature_type();
        let stun_exp = self.state.read_stun_expires_at_tick();
        let now = self.state.tick();
        // Roll previous → current cache.
        self.prev_hp.copy_from_slice(&self.last_hp);
        self.prev_stun_expires.copy_from_slice(&self.last_stun_expires);

        for slot in 0..(AGENT_COUNT as usize) {
            self.last_hp[slot] = hp.get(slot).copied().unwrap_or(0.0);
            self.last_alive[slot] = alive.get(slot).copied().unwrap_or(0);
            self.last_creature_type[slot] = creature_type.get(slot).copied().unwrap_or(0);
            self.last_stun_expires[slot] = stun_exp.get(slot).copied().unwrap_or(0);
            if self.last_hp[slot] > self.last_max_hp[slot] {
                self.last_max_hp[slot] = self.last_hp[slot];
            }
            let max = self.last_max_hp[slot].max(1.0);
            let frac = self.last_hp[slot] / max;
            let mut mat = material_for(self.last_creature_type[slot], frac);
            if self.is_stunned(slot) {
                mat = STUN_MATERIAL;
            }
            self.last_material[slot] = mat;
        }

        // Effect detection: compare per-slot HP delta + stun
        // transition. Damage/heal threshold is 0.001 to ignore
        // float noise. Push one marker per slot per tick.
        let expires = now + EFFECT_DURATION_TICKS;
        for slot in 0..(AGENT_COUNT as usize) {
            // Skip slots that just died this tick — the despawn
            // handles the visual; a damage flash on top of empty
            // space looks weird.
            if self.last_alive[slot] == 0 {
                continue;
            }
            let dhp = self.last_hp[slot] - self.prev_hp[slot];
            if dhp < -0.001 {
                self.effects.push(EffectMarker {
                    slot,
                    kind: EffectKind::Damage,
                    expires_at_tick: expires,
                });
            } else if dhp > 0.001 {
                self.effects.push(EffectMarker {
                    slot,
                    kind: EffectKind::Heal,
                    expires_at_tick: expires,
                });
            }
            // New stun: prev expiry was in the past, current is
            // in the future.
            let was_stunned = (self.prev_stun_expires[slot] as u64) > now.saturating_sub(1);
            let is_stunned_now = (self.last_stun_expires[slot] as u64) > now;
            if !was_stunned && is_stunned_now {
                self.effects.push(EffectMarker {
                    slot,
                    kind: EffectKind::Stun,
                    expires_at_tick: expires,
                });
            }
        }

        // Prune expired effects.
        self.effects.retain(|m| m.expires_at_tick > now);
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
        for slot in 0..(AGENT_COUNT as usize) {
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
        self.refresh_snapshot();
        // Termination heuristic: boss dead OR all heroes dead.
        let boss_dead = !self.boss_alive();
        let party_wiped = self.party_alive_count() == 0;
        if (boss_dead || party_wiped) && self.terminated_at_tick.is_none() {
            self.terminated_at_tick = Some(self.state.tick());
        }
        for slot in 0..(AGENT_COUNT as usize) {
            self.sync_slot(
                scene,
                slot,
                self.last_positions[slot],
                self.last_alive[slot] != 0,
                self.last_material[slot],
            );
        }
    }

    fn on_input(&mut self, _scene: &mut Scene, _event: &winit::event::WindowEvent) {}
}

/// Boss world position (after sim → voxel axis swap), exposed for
/// the bridge's stationary "objective" splat. Re-uses the existing
/// `objective_world_position`/`objective_material` symbols the
/// bridge expects from earlier pilots — boss_fight has no
/// objective marker concept, so we point them at the boss itself.
pub fn objective_world_position() -> Vec3 {
    Vec3::new(BOSS_POSITION.x, BOSS_POSITION.z, BOSS_POSITION.y)
}
pub fn objective_material() -> u8 {
    UNKNOWN_MATERIAL // unused by HUD; kept ABI-compatible for the bridge
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

        let post_setup = app.populated_entity_count();
        assert_eq!(
            post_setup,
            AGENT_COUNT as usize,
            "boss_fight spawns {} agents at construction; saw {}",
            AGENT_COUNT,
            post_setup,
        );

        for _ in 0..10 {
            voxel_engine::app::App::tick(&mut app, &mut scene, 0.1);
            if app.terminated_at_tick.is_some() {
                break;
            }
        }
        assert!(app.sim_tick() <= 10);
    }
}
