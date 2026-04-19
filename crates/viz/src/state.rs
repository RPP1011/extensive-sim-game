//! App state. Templated after `src/world_sim/voxel_app.rs::AppState`
//! but far smaller: one GpuVoxelTexture, no chunk pool, no terrain
//! compute, no mega-chunk bookkeeping.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec3;
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use voxel_engine::vulkan::voxel_gpu::{self, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use winit::keyboard::KeyCode;
use winit::window::Window;

use viz::grid_paint::{clear_above_ground, grid_index_of, paint_ground_plane, GRID_SIDE};
use viz::palette::{build_palette_rgba, PAL_DEATH};

// Matches src/world_sim/constants.rs for visual consistency with legacy renderer.
pub const RENDER_WIDTH:  u32 =  480;
pub const RENDER_HEIGHT: u32 =  270;
pub const WINDOW_WIDTH:  u32 = 1280;
pub const WINDOW_HEIGHT: u32 =  720;

pub struct AppState {
    // Held to keep the window alive for the lifetime of the Vulkan surface.
    // Never read directly after construction — the swapchain owns the surface.
    #[allow(dead_code)]
    pub window:    Window,
    pub ctx:       VulkanContext,
    pub alloc:     VulkanAllocator,
    pub swapchain: SwapchainContext,
    pub renderer:  VoxelRenderer,

    pub camera:         FreeCamera,
    pub keys_held:      HashSet<KeyCode>,
    pub mouse_captured: bool,
    pub last_mouse:     Option<(f64, f64)>,

    pub grid:        VoxelGrid,
    pub gpu_texture: Option<GpuVoxelTexture>,

    pub sim:       engine::state::SimState,
    pub scratch:   engine::step::SimScratch,
    pub events:    engine::event::EventRing,
    pub cascade:   engine::cascade::CascadeRegistry,
    pub backend:   engine::policy::UtilityBackend,
    #[allow(dead_code)]
    pub agent_ids: Vec<engine::ids::AgentId>,

    pub tick_period:         f32,
    pub sim_accum:           f32,
    pub sim_speed:           f32,
    pub paused:              bool,
    pub max_ticks_per_frame: u32,

    pub overlays: viz::overlays::OverlayTracker,

    pub scenario_path:    std::path::PathBuf,
    pub hud_timer:        Instant,
    pub frames_since_hud: u32,

    pub last_frame:  Instant,
}

impl AppState {
    pub fn new(window: Window, scenario: viz::scenario::Scenario, scenario_path: std::path::PathBuf) -> Result<Self> {
        let ctx       = VulkanContext::new_with_surface_extensions(&window).context("Vulkan context")?;
        let alloc     = VulkanAllocator::new(&ctx).context("allocator")?;
        let mut swapchain = SwapchainContext::new(&ctx, &window).context("swapchain")?;
        let renderer  = VoxelRenderer::new(&ctx, RENDER_WIDTH, RENDER_HEIGHT).context("renderer")?;
        swapchain.prepare_blit_commands(&ctx, renderer.gbuffer_output_image(), RENDER_WIDTH, RENDER_HEIGHT)
            .context("prepare blit")?;
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.12, 1.0]);

        let mut grid = VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE);
        paint_ground_plane(&mut grid);

        // Expand cap to at least the scenario's agent count so a
        // miscounted cap can't silently refuse a spawn.
        let needed = scenario.agent.len().max(1) as u32;
        let cap = scenario.world.agent_cap.max(needed);
        let mut sim = engine::state::SimState::new(cap, scenario.world.seed);
        let scratch = engine::step::SimScratch::new(cap as usize);
        let events  = engine::event::EventRing::with_cap(4096);
        let cascade = engine::cascade::CascadeRegistry::new();
        let backend = engine::policy::UtilityBackend;

        let mut agent_ids = Vec::with_capacity(scenario.agent.len());
        for spec in &scenario.agent {
            let ct = spec.creature()?;
            let spawn = engine::state::AgentSpawn { creature_type: ct, pos: spec.position(), hp: spec.hp };
            match sim.spawn_agent(spawn) {
                Some(id) => agent_ids.push(id),
                None => anyhow::bail!("spawn_agent returned None — agent_cap {} exhausted", cap),
            }
        }
        eprintln!("[viz] Spawned {} agents (cap {})", agent_ids.len(), cap);

        // Frame the whole spawn AABB at t=0. Previously we offset eye by a
        // constant 30 m regardless of spawn radius, which put small
        // scenarios on-camera but clipped large ones and vice-versa.
        //
        // Algorithm (Plan 3.1 Task 2):
        //   centroid = (min + max) / 2
        //   span     = |max - min|
        //   eye      = centroid + (0, span*1.5, span*1.5)   // above + back
        //   look_at  = centroid                              // straight at group
        //
        // `span * 1.5` keeps the whole AABB inside the frustum for FOV=π/4
        // with comfortable margin. A minimum span (MIN_FRAMING_SPAN) prevents
        // a single-agent or coincident-spawn scenario from collapsing the
        // camera onto the subject.
        const MIN_FRAMING_SPAN: f32 = 20.0;
        let (centroid, span): (Vec3, f32) = if scenario.agent.is_empty() {
            let c = Vec3::new((GRID_SIDE as f32) * 0.5, 2.0, (GRID_SIDE as f32) * 0.5);
            (c, MIN_FRAMING_SPAN)
        } else {
            let first = scenario.agent[0].position();
            let (mn, mx) = scenario.agent.iter().fold((first, first), |(mn, mx), a| {
                let p = a.position();
                (mn.min(p), mx.max(p))
            });
            let c = (mn + mx) * 0.5;
            let s = mn.distance(mx).max(MIN_FRAMING_SPAN);
            (c, s)
        };
        let eye = centroid + Vec3::new(0.0, span * 1.5, span * 1.5);
        let mut camera = FreeCamera::new(eye, centroid);
        camera.set_move_speed(20.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            keys_held: HashSet::new(),
            mouse_captured: false, last_mouse: None,
            grid, gpu_texture: None,
            sim, scratch, events, cascade, backend, agent_ids,
            tick_period: 0.1,
            sim_accum:   0.0,
            sim_speed:   1.0,
            paused:      false,
            max_ticks_per_frame: 8,
            overlays: viz::overlays::OverlayTracker::new(),
            scenario_path,
            hud_timer: Instant::now(),
            frames_since_hud: 0,
            last_frame: Instant::now(),
        })
    }

    /// Reset sim state from a fresh `Scenario`. Preserves Vulkan
    /// resources + camera pose; resets sim_accum / sim_speed / paused
    /// to their default values. Overlays are dropped.
    pub fn reload_scenario(&mut self, scenario: viz::scenario::Scenario) -> Result<()> {
        let needed = scenario.agent.len().max(1) as u32;
        let cap = scenario.world.agent_cap.max(needed);
        let mut sim = engine::state::SimState::new(cap, scenario.world.seed);
        let scratch = engine::step::SimScratch::new(cap as usize);
        let events  = engine::event::EventRing::with_cap(4096);
        let cascade = engine::cascade::CascadeRegistry::new();
        let backend = engine::policy::UtilityBackend;

        let mut agent_ids = Vec::with_capacity(scenario.agent.len());
        for spec in &scenario.agent {
            let ct = spec.creature()?;
            let spawn = engine::state::AgentSpawn { creature_type: ct, pos: spec.position(), hp: spec.hp };
            match sim.spawn_agent(spawn) {
                Some(id) => agent_ids.push(id),
                None => anyhow::bail!("spawn_agent returned None — cap {} exhausted", cap),
            }
        }

        self.sim = sim;
        self.scratch = scratch;
        self.events = events;
        self.cascade = cascade;
        self.backend = backend;
        self.agent_ids = agent_ids;
        self.overlays = viz::overlays::OverlayTracker::new();
        self.sim_accum = 0.0;
        self.sim_speed = 1.0;
        self.paused = false;
        Ok(())
    }

    /// Advance the sim as many ticks as accumulated `dt` allows. Caps at
    /// `max_ticks_per_frame` so a long hitch doesn't spool up a burst.
    /// Returns the number of ticks executed. Zero when paused.
    pub fn tick_sim(&mut self, dt: f32) -> u32 {
        if self.paused { return 0; }
        let effective_period = self.tick_period / self.sim_speed.max(0.001);
        self.sim_accum += dt;
        let mut ticked = 0;
        while self.sim_accum >= effective_period && ticked < self.max_ticks_per_frame {
            engine::step::step(
                &mut self.sim, &mut self.scratch, &mut self.events,
                &self.backend, &self.cascade,
            );
            self.sim_accum -= effective_period;
            ticked += 1;
        }
        if self.sim_accum > effective_period * (self.max_ticks_per_frame as f32) {
            self.sim_accum = 0.0; // discard excess to prevent runaway
        }
        ticked
    }

    pub fn upload_grid(&mut self) -> Result<()> {
        let palette = build_palette_rgba();
        let new_tex = voxel_gpu::upload_grid_to_gpu(
            &self.ctx, &mut self.alloc, &self.grid, &palette,
        ).context("upload voxel grid")?;
        if let Some(old) = self.gpu_texture.take() {
            old.destroy(&self.ctx, &mut self.alloc);
        }
        self.gpu_texture = Some(new_tex);
        Ok(())
    }

    /// One rendered frame via `render_frame_gpu` + `present_blit`.
    /// `render_frame_gpu` CPU-waits on its fence internally so the
    /// subsequent blit can read the gbuffer RT without a semaphore.
    #[allow(clippy::type_complexity)]
    pub fn render(&mut self) -> Result<()> {
        clear_above_ground(&mut self.grid);

        self.overlays.ingest_with_state(&self.events, &self.sim);
        self.overlays.prune(self.sim.tick);

        // -----------------------------------------------------------
        // Vertical stacking workaround (Plan 3.1 Task 3).
        //
        // The engine has NO collision detection — multiple agents can
        // (and frequently do) occupy the same Vec3 position. When we
        // paint each agent into a single voxel cell they overwrite each
        // other and the render visually collapses several agents into
        // one cube, hiding casualties of that bug. We paper over this
        // presentationally by bucketing agents by their (x, z) voxel
        // index and stacking them on the Y axis — alive agents first
        // at the base height, then death markers above — so every
        // agent + death overlay is visible as a distinct cube even when
        // the underlying positions overlap.
        //
        // This is PURELY VISUAL. SimState positions are unchanged. The
        // real fix (a collision pass in the tick pipeline) is tracked
        // separately; see `docs/engine/status.md`.
        // -----------------------------------------------------------
        // (x, z) → [(palette_idx, base_y)] for alive agents.
        let mut alive_by_cell: HashMap<(u32, u32), Vec<(u8, u32)>> = HashMap::new();
        for id in self.sim.agents_alive() {
            let pos = match self.sim.agent_pos(id) { Some(p) => p, None => continue };
            let ct  = self.sim.agent_creature_type(id).unwrap_or(engine::creature::CreatureType::Human);
            let idx = viz::palette::creature_palette_index(ct);
            let lifted = Vec3::new(
                pos.x,
                pos.y.max((viz::grid_paint::GROUND_Y + 1) as f32),
                pos.z,
            );
            if let Some((x, y, z)) = grid_index_of(lifted, &self.grid) {
                alive_by_cell.entry((x, z)).or_default().push((idx, y));
            }
        }
        // Bucket death markers by the same (x, z) grid so they can stack
        // on top of any alive agents (or other deaths) sharing the cell.
        let mut deaths_by_cell: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for at in self.overlays.death_positions() {
            let lifted = Vec3::new(
                at.x,
                at.y.max((viz::grid_paint::GROUND_Y + 1) as f32),
                at.z,
            );
            if let Some((x, y, z)) = grid_index_of(lifted, &self.grid) {
                deaths_by_cell.entry((x, z)).or_default().push(y);
            }
        }
        // Paint alive agents at base y, then death markers stacked above.
        let grid_h = self.grid.dimensions().1;
        for ((x, z), stack) in &alive_by_cell {
            for (offset, (idx, base_y)) in stack.iter().enumerate() {
                let y = (*base_y).saturating_add(offset as u32);
                if y < grid_h {
                    self.grid.set(*x, y, *z, *idx);
                }
            }
        }
        for ((x, z), deaths) in &deaths_by_cell {
            let alive_count = alive_by_cell.get(&(*x, *z)).map_or(0, |v| v.len()) as u32;
            // Use the lowest death marker's intended y as the base; this
            // handles the rare case of a death at a cell with no alive
            // tenants correctly.
            let base_y = deaths.iter().copied().min().unwrap_or(0);
            for (offset, _) in deaths.iter().enumerate() {
                let y = base_y.saturating_add(alive_count).saturating_add(offset as u32);
                if y < grid_h {
                    self.grid.set(*x, y, *z, PAL_DEATH);
                }
            }
        }

        // Attack lines + announce rings. Death markers are handled
        // above as part of the stacking pass.
        self.overlays.paint_non_death(&mut self.grid, self.sim.tick);

        self.upload_grid()?;
        let tex = self.gpu_texture.as_ref().expect("upload_grid always populates gpu_texture");
        let dims = [GRID_SIDE as f32; 3];
        let position = [0.0f32, 0.0, 0.0];
        let palette_color = [1.0f32, 1.0, 1.0, 1.0];
        let objects: [(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3]); 1] =
            [(tex, palette_color, position, dims)];
        self.renderer.render_frame_gpu(&self.ctx, &self.camera, &objects)
            .context("render_frame_gpu")?;
        let src = self.renderer.gbuffer_output_image();
        self.swapchain.present_blit(&self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT)
            .context("present_blit")?;
        Ok(())
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed { self.keys_held.insert(key); } else { self.keys_held.remove(&key); }
        if !pressed { return; }
        match key {
            KeyCode::Space => {
                self.paused = !self.paused;
                eprintln!("[viz] {}", if self.paused { "PAUSED" } else { "RUNNING" });
            }
            KeyCode::Period => {
                if self.paused {
                    engine::step::step(
                        &mut self.sim, &mut self.scratch, &mut self.events,
                        &self.backend, &self.cascade,
                    );
                    eprintln!("[viz] step → tick {}", self.sim.tick);
                } else {
                    eprintln!("[viz] '.' only steps while paused");
                }
            }
            KeyCode::KeyR => {
                match viz::scenario::load(&self.scenario_path) {
                    Ok(sc) => match self.reload_scenario(sc) {
                        Ok(()) => eprintln!("[viz] reloaded {:?}", self.scenario_path),
                        Err(e) => eprintln!("[viz] reload failed: {}", e),
                    },
                    Err(e) => eprintln!("[viz] reload read failed: {}", e),
                }
            }
            KeyCode::BracketLeft  => {
                self.sim_speed = (self.sim_speed * 0.5).max(0.0625);
                eprintln!("[viz] speed {:.3}x", self.sim_speed);
            }
            KeyCode::BracketRight => {
                self.sim_speed = (self.sim_speed * 2.0).min(32.0);
                eprintln!("[viz] speed {:.3}x", self.sim_speed);
            }
            _ => {}
        }
    }

    pub fn maybe_emit_hud(&mut self) {
        self.frames_since_hud += 1;
        let elapsed = self.hud_timer.elapsed().as_secs_f32();
        if elapsed < 1.0 { return; }
        let fps = (self.frames_since_hud as f32) / elapsed;
        let alive = self.sim.agents_alive().count();
        let state_str = if self.paused { "PAUSED" } else { "RUNNING" };
        eprintln!(
            "[hud] {} tick={} alive={}/{} speed={:.2}x fps={:.0} overlays={}",
            state_str, self.sim.tick, alive, self.agent_ids.len(),
            self.sim_speed, fps, self.overlays.len(),
        );
        self.hud_timer = Instant::now();
        self.frames_since_hud = 0;
    }

    pub fn update_camera(&mut self, dt: f32) {
        use voxel_engine::camera::{CameraController, InputState};
        if self.keys_held.is_empty() { return; }
        let forward = if self.keys_held.contains(&KeyCode::KeyW) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyS) { -1.0 } else { 0.0 };
        let right = if self.keys_held.contains(&KeyCode::KeyD) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyA) { -1.0 } else { 0.0 };
        let up = if self.keys_held.contains(&KeyCode::KeyE) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyQ) { -1.0 } else { 0.0 };
        self.camera.update(&InputState {
            move_forward: forward, move_right: right, move_up: up,
            mouse_dx: 0.0, mouse_dy: 0.0, scroll_delta: 0.0,
        }, dt);
    }

    pub fn run_frame(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_camera(dt);
        let _n = self.tick_sim(dt);
        if let Err(e) = self.render() { eprintln!("[viz] render error: {}", e); }
        self.maybe_emit_hud();
    }
}
