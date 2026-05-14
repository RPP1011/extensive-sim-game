//! `viewer_app` — windowed viewer for `sims::dungeon_horde`.
//!
//! Opens a winit window, drives voxel_engine's Vulkan renderer, and
//! ticks the sim at 100ms wall-clock cadence. Top-down camera framed
//! on the dungeon centroid.
//!
//! Usage:
//!
//! ```text
//! cargo run -p viewer_runtime --bin viewer_app --release [SEED]
//! ```
//!
//! With no SEED, defaults to `0xD007_BEEF_5_7EA1` — the same seed the
//! `dungeon_horde_pin` test uses, so the rendered scene matches the
//! pinned test layout.
//!
//! What you should see:
//!
//! - A top-down view of a multi-room voxel dungeon (6×6 grid of room
//!   slots, ~22 rooms connected via doorways).
//! - Gray walls (stone) outlining the rooms; tan floors inside.
//! - Five hero cubes (warrior=brown, cleric=white, ranger=green,
//!   mage=blue, rogue=purple) at the spawn-room centroid.
//! - Hundreds of enemy cubes scattered through the rooms: orange
//!   archers, dim-green goblins, dark-red brutes.
//! - Per-tick (100ms wall clock) the agents move toward each other
//!   and start fighting; dead agents disappear from the scene.

use std::sync::Arc;
use std::time::{Duration, Instant};

use viewer_runtime::{ViewerApp, VoxelBridge, BRIDGE_DIM_X, BRIDGE_DIM_Z};
use viewer_runtime::mesh_renderer::{InstanceData, MeshCpu, MeshDraw, MeshRendererGpu};
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Wall-clock period between sim ticks. The .sim's `init { hp, etc }` and
/// ability cooldowns are in *sim* ticks, so halving this constant just
/// plays the same deterministic sim back 2x faster — the auto-restart
/// reel cycles through seeds in ~60-80s wall clock instead of ~120-160s.
const SIM_TICK_PERIOD: Duration = Duration::from_millis(50);
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Default seed — same as `dungeon_horde_pin.rs::SEED_U64` so the
/// rendered scene matches the pinned test layout.
const DEFAULT_SEED: u64 = 0xD007_BEEF_5_7EA1;

/// Top-down observer camera, framed on the dungeon centroid. The
/// dungeon is `GRID_X × GRID_Y` cells covering world `[0, GRID_X) × [0, GRID_Y)`,
/// so the centroid is at `(GRID_X/2, GRID_Y/2, 1)`. Camera height
/// (Y, since voxel_engine is Y-up) is enough to frame the whole
/// dungeon vertically.
fn observer_camera() -> FreeCamera {
    // Bridge grid is (X horizontal, Y vertical, Z depth) post axis-swap.
    // Centroid in the X-Z plane; camera sits above on the Y axis.
    let cx = BRIDGE_DIM_X as f32 / 2.0;
    let cz = BRIDGE_DIM_Z as f32 / 2.0;
    // 3/4 isometric-ish view: eye is pulled back ~0.7× height along +Z and
    // the look-at target is lifted slightly off the floor so character
    // meshes (~1 unit tall, planted at Y=1) sit mid-frame instead of
    // being squashed at the bottom of a steep top-down. Tilt of ~35° off
    // vertical gives a clear read of model 3D shape while still showing
    // the dungeon layout.
    let height = BRIDGE_DIM_X.max(BRIDGE_DIM_Z) as f32 + 8.0;
    FreeCamera::new(
        glam::Vec3::new(cx, height, cz + height * 0.7),
        glam::Vec3::new(cx, 1.5, cz),
    )
}

struct WindowedViewer {
    seed: u64,
    app: ViewerApp,
    camera: FreeCamera,
    last_tick: Instant,
    /// Constructed lazily on first `resumed()` — winit 0.30 doesn't
    /// give a window until the event loop is running.
    gfx: Option<Gfx>,
    /// Wall-clock instant at which the current run terminated.
    /// `None` while the run is still in progress. Used to delay
    /// auto-restart so the user can read the verdict + see the
    /// outcome tint before the next dungeon rolls in.
    terminated_at_wall: Option<Instant>,
    /// Pause flag — toggled by Space. While true, sim ticks don't
    /// advance and auto-restart is suppressed; rendering still
    /// happens so the user can study the frozen frame.
    paused: bool,
    /// Total runs completed in this session (incremented on each
    /// auto-restart). Drives the cross-run aggregate line.
    session_runs: u32,
    /// Subset of `session_runs` that ended in DUNGEON CLEARED.
    session_wins: u32,
    /// Sum of `app.sim_tick()` at termination across all completed
    /// runs. Divided by `session_runs` for the mean-ticks metric.
    session_tick_total: u64,
}

struct Gfx {
    window: Arc<Window>,
    ctx: VulkanContext,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
    bridge: VoxelBridge,
    /// Mesh pass: draws agent character models over the voxel output
    /// via `present_blit_with_overlay`. `None` when mesh load failed
    /// at startup (renderer still works, just no meshes painted).
    mesh: Option<MeshRendererGpu>,
    /// Per-role mesh slot indices into `mesh.meshes`.
    /// `hero_slots[role - 1]` for heroes.
    hero_slots: [Option<usize>; 5],
    /// Two mesh variants per enemy creature type for visual variety.
    /// `enemy_slots[ct * 2 + (agent_slot % 2)]` selects the variant.
    /// Indices 0/1 = Archer, 2/3 = Brute, 4/5 = Goblin.
    enemy_slots: [Option<usize>; 6],
}

impl ApplicationHandler for WindowedViewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.title_for_tick(0))
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_W, WINDOW_H));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create_window failed"),
        );

        let ctx = VulkanContext::new_with_surface_extensions(&window)
            .expect("VulkanContext::new_with_surface_extensions failed");
        let swapchain = SwapchainContext::new(&ctx, &window)
            .expect("SwapchainContext::new failed");
        let renderer = VoxelRenderer::new(&ctx, WINDOW_W, WINDOW_H)
            .expect("VoxelRenderer::new failed");

        let mut bridge =
            VoxelBridge::new(&ctx, &self.app).expect("VoxelBridge::new failed");
        bridge
            .refresh(&ctx, &self.app)
            .expect("VoxelBridge::refresh (initial) failed");

        // Construct the mesh-pass overlay + load per-role / per-creature
        // glTF meshes from Kenney's mini-characters pack. Order matters:
        // glTF loaders prefer their own thread-local Vulkan contexts;
        // build the renderer first, then add_mesh for each slot.
        let mut mesh = match MeshRendererGpu::new(
            &ctx,
            swapchain.image_views(),
            swapchain.extent(),
            swapchain.surface_format(),
            /*max_instances=*/ 1024,
        ) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("[viewer_app] MeshRendererGpu::new failed: {e:#}");
                None
            }
        };

        const PACK: &str = "assets/models/kenney_mini-characters/Models/GLB format";
        // Role → mesh assignments. Heroes get the male-a..e palette;
        // enemies are split across remaining variants for visual variety.
        let hero_files = [
            "character-male-a.glb",   // Warrior (role=1)
            "character-female-a.glb", // Cleric  (role=2)
            "character-male-b.glb",   // Ranger  (role=3)
            "character-female-b.glb", // Mage    (role=4)
            "character-male-c.glb",   // Rogue   (role=5)
        ];
        let enemy_files = [
            "character-male-d.glb",   // Archer A
            "character-female-d.glb", // Archer B
            "character-male-f.glb",   // Brute  A
            "character-male-e.glb",   // Brute  B
            "character-female-c.glb", // Goblin A
            "character-female-e.glb", // Goblin B
        ];

        let load_slot = |mesh: &mut Option<MeshRendererGpu>, file: &str| -> Option<usize> {
            let path = format!("{PACK}/{file}");
            match MeshCpu::load_glb(&path) {
                Ok(m) => match mesh.as_mut()?.add_mesh(&ctx, &m) {
                    Ok(slot) => {
                        eprintln!(
                            "[viewer_app] mesh slot {slot}: {file} ({} verts, {} tris)",
                            m.vertex_count(), m.triangle_count(),
                        );
                        Some(slot)
                    }
                    Err(e) => {
                        eprintln!("[viewer_app] add_mesh({file}) failed: {e:#}");
                        None
                    }
                },
                Err(e) => {
                    eprintln!("[viewer_app] load_glb({file}) failed: {e:#}");
                    None
                }
            }
        };
        let mut hero_slots: [Option<usize>; 5] = [None; 5];
        for (i, file) in hero_files.iter().enumerate() {
            hero_slots[i] = load_slot(&mut mesh, file);
        }
        let mut enemy_slots: [Option<usize>; 6] = [None; 6];
        for (i, file) in enemy_files.iter().enumerate() {
            enemy_slots[i] = load_slot(&mut mesh, file);
        }

        window.request_redraw();

        self.gfx = Some(Gfx {
            window, ctx, swapchain, renderer, bridge, mesh,
            hero_slots, enemy_slots,
        });
    }

    /// Drive continuous animation. winit calls about_to_wait whenever
    /// the event queue drains; without an explicit request_redraw
    /// here, RedrawRequested fires only on OS-driven invalidation.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, repeat: false, .. },
                ..
            } => {
                match logical_key {
                    Key::Named(NamedKey::Space) => {
                        self.paused = !self.paused;
                        eprintln!(
                            "[viewer_app] {} @ tick {}",
                            if self.paused { "PAUSED" } else { "RESUMED" },
                            self.app.sim_tick(),
                        );
                    }
                    Key::Character(c) if c.eq_ignore_ascii_case("r") => {
                        // Manual reset — jump to the next seed immediately
                        // without waiting for natural termination + hold.
                        let next_seed = self.seed.wrapping_add(1);
                        eprintln!(
                            "[viewer_app] manual reset (R): seed 0x{:X} -> 0x{:X}",
                            self.seed, next_seed,
                        );
                        if let Some(new_app) = ViewerApp::try_new(next_seed) {
                            self.app = new_app;
                            self.seed = next_seed;
                            self.last_tick = Instant::now();
                            self.terminated_at_wall = None;
                            self.paused = false;
                            if let Some(gfx) = self.gfx.as_mut() {
                                let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                                let old_bridge = std::mem::replace(
                                    &mut gfx.bridge,
                                    VoxelBridge::new(&gfx.ctx, &self.app)
                                        .expect("VoxelBridge::new (manual reset) failed"),
                                );
                                old_bridge.destroy(&gfx.ctx);
                                if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                                    eprintln!("[viewer_app] post-reset refresh failed: {e}");
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CloseRequested => {
                eprintln!(
                    "[viewer_app] CloseRequested — exit at tick={}",
                    self.app.sim_tick(),
                );
                if self.session_runs > 0 {
                    let mean_ticks = self.session_tick_total / self.session_runs as u64;
                    eprintln!(
                        "[viewer_app] final session: {} runs ({} cleared, {} TPK, mean {} ticks, win-rate {:.0}%)",
                        self.session_runs,
                        self.session_wins,
                        self.session_runs - self.session_wins,
                        mean_ticks,
                        100.0 * (self.session_wins as f32) / (self.session_runs as f32),
                    );
                }
                if let Some(mut gfx) = self.gfx.take() {
                    let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                    if let Some(mut mesh) = gfx.mesh.take() { mesh.destroy(&gfx.ctx); }
                    gfx.bridge.destroy(&gfx.ctx);
                    gfx.swapchain.destroy(&gfx.ctx);
                    gfx.renderer.destroy(&gfx.ctx);
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Catch up on fixed-step sim ticks. Bound at 8 to
                // avoid runaway after a long pause. Pause flag freezes
                // sim advancement but keeps render+present going.
                let mut ticks_this_frame: u32 = 0;
                if self.paused {
                    // Slide last_tick forward so resuming doesn't trigger
                    // a flood of catch-up ticks (would burst-advance the
                    // sim by however long the pause lasted).
                    self.last_tick = Instant::now();
                }
                while !self.paused && self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks_this_frame < 8 {
                    self.app.step();
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                    if let Some(gfx) = self.gfx.as_mut() {
                        if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                            eprintln!("[viewer_app] VoxelBridge::refresh failed: {e}");
                        }
                    }
                }

                // Auto-restart with the next seed once the user has had a
                // few seconds to read the outcome. Bumping the seed gives
                // a fresh dungeon roll without the user having to relaunch
                // the binary — the viewer becomes a continuous demo reel.
                const POST_TERMINATION_HOLD: Duration = Duration::from_secs(3);
                if !self.paused && self.app.terminated_at_tick.is_some() && self.terminated_at_wall.is_none() {
                    self.terminated_at_wall = Some(Instant::now());
                }
                if let Some(t) = self.terminated_at_wall {
                    if !self.paused && t.elapsed() >= POST_TERMINATION_HOLD {
                        // Capture this run's contribution to the session aggregate
                        // before we tear it down. `outcome` is Some by construction
                        // here (we only enter the hold path after termination).
                        let won = self.app.outcome.unwrap_or(false);
                        self.session_runs += 1;
                        if won { self.session_wins += 1; }
                        self.session_tick_total += self.app.sim_tick();
                        let mean_ticks = self.session_tick_total / self.session_runs as u64;
                        eprintln!(
                            "[viewer_app] session: {} runs ({} cleared, {} TPK, mean {} ticks)",
                            self.session_runs,
                            self.session_wins,
                            self.session_runs - self.session_wins,
                            mean_ticks,
                        );

                        let next_seed = self.seed.wrapping_add(1);
                        eprintln!(
                            "[viewer_app] auto-restart: seed 0x{:X} -> 0x{:X}",
                            self.seed, next_seed,
                        );
                        if let Some(new_app) = ViewerApp::try_new(next_seed) {
                            self.app = new_app;
                            self.seed = next_seed;
                            self.last_tick = Instant::now();
                            self.terminated_at_wall = None;
                            // Rebuild the bridge so the dungeon walls/floor
                            // get re-uploaded from the new ViewerApp's grid.
                            if let Some(gfx) = self.gfx.as_mut() {
                                let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                                let old_bridge = std::mem::replace(
                                    &mut gfx.bridge,
                                    VoxelBridge::new(&gfx.ctx, &self.app)
                                        .expect("VoxelBridge::new (auto-restart) failed"),
                                );
                                old_bridge.destroy(&gfx.ctx);
                                if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                                    eprintln!("[viewer_app] post-restart refresh failed: {e}");
                                }
                            }
                        } else {
                            eprintln!(
                                "[viewer_app] auto-restart failed (no wgpu adapter on next try).                                  Holding on the current run."
                            );
                            // Push the hold deadline forward so we don't hammer the failure path.
                            self.terminated_at_wall = Some(Instant::now());
                        }
                    }
                }
                let title = self.title_for_tick(self.app.sim_tick());
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);
                    let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
                    if let Err(e) = gfx
                        .renderer
                        .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                    {
                        eprintln!("[viewer_app] render_frame_gpu failed: {e}");
                        return;
                    }
                    // Bucket agents by mesh slot so we can do one draw per
                    // mesh with first_instance offsets. 8 buckets: 5 hero
                    // roles + 3 enemy creature types. Any agent whose slot
                    // is None (mesh load failed) is silently skipped.
                    let mut buckets: [Vec<InstanceData>; 11] =
                        std::array::from_fn(|_| Vec::new());
                    let mut bucket_slot: [Option<usize>; 11] = [None; 11];
                    for i in 0..5 { bucket_slot[i] = gfx.hero_slots[i]; }
                    for i in 0..6 { bucket_slot[5 + i] = gfx.enemy_slots[i]; }
                    if gfx.mesh.is_some() {
                        // Tick interpolation alpha: 0 at start of current
                        // tick, 1 just before next tick. Smooths agent motion
                        // between 20 Hz sim ticks.
                        let interp_alpha = (self.last_tick.elapsed().as_secs_f32()
                            / SIM_TICK_PERIOD.as_secs_f32()).clamp(0.0, 1.0);
                        for (idx, agent) in self.app.agents().iter().enumerate() {
                            let death_tick = self.app.death_ticks[idx];
                            let is_dying = !agent.alive
                                && death_tick > 0
                                && death_tick < viewer_runtime::DEATH_FADE_TICKS;
                            if !agent.alive && !is_dying { continue; }
                            // Lerp sim-space position between previous and
                            // current tick using interp_alpha. prev_positions
                            // is [x, y, z] in sim space (Z-up).
                            let prev = self.app.prev_positions[idx];
                            let sx = prev[0] + (agent.pos.x - prev[0]) * interp_alpha;
                            let sy = prev[1] + (agent.pos.y - prev[1]) * interp_alpha;
                            let sz = prev[2] + (agent.pos.z - prev[2]) * interp_alpha;
                            // sim (x, y, z) -> renderer (x, z, y) — same swap
                            // the voxel bridge applies.
                            let mesh_pos = glam::Vec3::new(sx, sz, sy);
                            let tint = tint_for_agent(agent.creature_type, agent.role);
                            // Per-creature-type scale gives visible size
                            // hierarchy: goblin < archer < hero < brute < BOSS.
                            // Boss = matches the boss_slot the bridge
                            // already tracks; gets a dramatic 2x.
                            let base_scale = if Some(idx as u32) == self.app.boss_slot {
                                1.8
                            } else if agent.creature_type == 3 /*HERO*/ {
                                0.85
                            } else if agent.creature_type == 1 /*BRUTE*/ {
                                1.1
                            } else if agent.creature_type == 0 /*ARCHER*/ {
                                0.80
                            } else if agent.creature_type == 2 /*GOBLIN*/ {
                                0.65
                            } else {
                                0.75
                            };
                            // Death fade: scale shrinks from 1.0 → 0.0 over
                            // DEATH_FADE_TICKS, tint dims to 40%.
                            let (scale, tint_mul) = if is_dying {
                                let t = death_tick as f32
                                    / viewer_runtime::DEATH_FADE_TICKS as f32;
                                (base_scale * (1.0 - t).max(0.0), 0.4)
                            } else {
                                (base_scale, 1.0)
                            };
                            // Heal flash: blend tint toward MAT_HEAL (mint
                            // green) for a few frames after HP increases.
                            // Damage flash on the voxel splat is enough —
                            // we don't double-flash the mesh to keep combat
                            // hits as the highest-signal visual event.
                            let healing = self.app.heal_ticks[idx] > 0;
                            let tint = if healing {
                                [0.7, 1.0, 0.78]
                            } else {
                                [
                                    tint[0] * tint_mul,
                                    tint[1] * tint_mul,
                                    tint[2] * tint_mul,
                                ]
                            };
                            // Combat facing override: heroes face their
                            // nearest live enemy within 18 units (longest
                            // hero ability range) instead of just movement
                            // direction. Makes stationary firing visibly
                            // aim at targets. Enemies keep movement facing.
                            let facing_xy = if agent.creature_type == 3 /*HERO*/ {
                                const ATTACK_RANGE_SQ: f32 = 18.0 * 18.0;
                                let mut best_sq = f32::MAX;
                                let mut best: Option<[f32; 2]> = None;
                                for other in self.app.agents() {
                                    if !other.alive { continue; }
                                    if other.creature_type == 3 /*HERO*/ { continue; }
                                    let dx = other.pos.x - agent.pos.x;
                                    let dy = other.pos.y - agent.pos.y;
                                    let d2 = dx*dx + dy*dy;
                                    if d2 < best_sq && d2 < ATTACK_RANGE_SQ && d2 > 1e-6 {
                                        best_sq = d2;
                                        let inv = 1.0 / d2.sqrt();
                                        best = Some([dx * inv, dy * inv]);
                                    }
                                }
                                best.unwrap_or(agent.facing_xy)
                            } else {
                                agent.facing_xy
                            };
                            // Sim facing is XY (sim's horizontal plane);
                            // renderer's horizontal plane is XZ — same Z↔Y swap
                            // applies to the heading vector.
                            let facing_renderer = [
                                facing_xy[0],
                                0.0,
                                facing_xy[1],
                            ];
                            let inst = InstanceData::from_pos_facing(
                                mesh_pos, scale, facing_renderer, tint,
                            );
                            // Heroes go to role-indexed bucket (0..5); enemies
                            // pick between 2 variants per type by (idx % 2):
                            //   creature_type=0 (Archer) → bucket 5 or 6
                            //   creature_type=1 (Brute)  → bucket 7 or 8
                            //   creature_type=2 (Goblin) → bucket 9 or 10
                            let bucket = if agent.creature_type == 3 /*HERO*/ {
                                ((agent.role as i32) - 1).clamp(0, 4) as usize
                            } else if agent.creature_type <= 2 {
                                5 + (agent.creature_type as usize) * 2 + (idx % 2)
                            } else { continue };
                            buckets[bucket].push(inst);
                        }
                    }
                    let draws: Vec<MeshDraw> = (0..11)
                        .filter_map(|b| {
                            let slot = bucket_slot[b]?;
                            if buckets[b].is_empty() { return None; }
                            Some(MeshDraw { mesh_slot: slot, instances: &buckets[b] })
                        })
                        .collect();
                    let view_proj = view_proj_for_camera(&self.camera);

                    let mesh_renderer_opt = gfx.mesh.as_mut();
                    let ctx_ref = &gfx.ctx;
                    let present_result = gfx.swapchain.present_blit_with_overlay(
                        ctx_ref,
                        gfx.renderer.light_output_image(),
                        WINDOW_W,
                        WINDOW_H,
                        ash::vk::Semaphore::null(),
                        |cmd, image_index| {
                            if let Some(mesh) = mesh_renderer_opt {
                                mesh.record_overlay(
                                    ctx_ref, cmd, image_index, view_proj, &draws,
                                )?;
                            }
                            Ok(())
                        },
                    );
                    if let Err(e) = present_result {
                        eprintln!("[viewer_app] present_blit_with_overlay failed: {e}");
                        return;
                    }
                }
            }
            _ => {}
        }
    }
}

impl WindowedViewer {
    fn title_for_tick(&self, tick: u64) -> String {
        let mut alive_h = 0u32;
        let mut alive_e = 0u32;
        for a in self.app.agents() {
            if !a.alive { continue; }
            if a.creature_type == viewer_runtime::dungeon::CT_HERO {
                alive_h += 1;
            } else {
                alive_e += 1;
            }
        }
        format!(
            "viewer_runtime — seed=0x{:X} tick={} heroes={}/5 enemies={}",
            self.seed, tick, alive_h, alive_e,
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| {
            if let Some(stripped) = s.strip_prefix("0x") {
                u64::from_str_radix(stripped, 16).ok()
            } else {
                s.parse().ok()
            }
        })
        .unwrap_or(DEFAULT_SEED);

    eprintln!(
        "[viewer_app] seed=0x{seed:X} — opening window {WINDOW_W}x{WINDOW_H}"
    );
    eprintln!(
        "[viewer_app] What you should see: top-down view of a voxel dungeon \
         (gray walls, tan floors, ~22 rooms). 5 hero cubes at spawn (brown=Warrior, \
         white=Cleric, green=Ranger, blue=Mage, purple=Rogue). Hundreds of enemy \
         cubes (orange=Archer, dim-green=Goblin, dark-red=Brute) in deeper rooms. \
         Tick advances every 100ms; agents move + die over time."
    );

    let app = match ViewerApp::try_new(seed) {
        Some(a) => a,
        None => {
            eprintln!(
                "[viewer_app] no compatible wgpu adapter found for sim runtime — \
                 cannot initialize. Try running on a host with a discrete GPU \
                 or software fallback (lavapipe, etc.)."
            );
            std::process::exit(2);
        }
    };
    let camera = observer_camera();

    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut viewer = WindowedViewer {
        seed,
        app,
        camera,
        last_tick: Instant::now(),
        gfx: None,
        terminated_at_wall: None,
        paused: false,
        session_runs: 0,
        session_wins: 0,
        session_tick_total: 0,
    };
    event_loop
        .run_app(&mut viewer)
        .expect("event_loop.run_app failed");
}


// ---------------------------------------------------------------------
// Mesh-pass helpers.
// ---------------------------------------------------------------------

/// Build the view-projection matrix the mesh pass uses for its
/// push constant. voxel_engine renders with its own camera math; the
/// mesh pass needs an equivalent VP so meshes line up with voxels.
///
/// FreeCamera exposes view (look_at) directly; we wrap with a
/// perspective projection at the same FOV the renderer uses
/// (~60° vertical) plus Vulkan-y Y-flip in clip space.
fn view_proj_for_camera(cam: &FreeCamera) -> glam::Mat4 {
    // Reuse FreeCamera's own view + projection matrices so the mesh
    // pass lines up exactly with voxel_engine's rendering. These are
    // returned as 16-element f32 arrays in column-major order, which is
    // also how Mat4::from_cols_array reads them.
    let view = glam::Mat4::from_cols_array(&cam.view_matrix_array());
    let aspect = WINDOW_W as f32 / WINDOW_H as f32;
    let proj = glam::Mat4::from_cols_array(&cam.projection_matrix_array(aspect));
    proj * view
}

/// Per-agent RGB tint (0..1) for the mesh pass. Mirrors the voxel
/// palette in `viewer_runtime::lib` so the mesh layer reads as the
/// same agent class as the splat-layer fallback. Heroes by role,
/// enemies by creature type.
fn tint_for_agent(creature_type: u32, role: u32) -> [f32; 3] {
    const CT_HERO: u32 = 3;
    const CT_ARCHER: u32 = 0;
    const CT_BRUTE: u32 = 1;
    const CT_GOBLIN: u32 = 2;
    if creature_type == CT_HERO {
        match role {
            1 => [140.0 / 255.0, 95.0 / 255.0, 50.0 / 255.0],   // Warrior — brown
            2 => [240.0 / 255.0, 240.0 / 255.0, 240.0 / 255.0], // Cleric  — white
            3 => [60.0 / 255.0, 170.0 / 255.0, 75.0 / 255.0],   // Ranger  — green
            4 => [70.0 / 255.0, 130.0 / 255.0, 230.0 / 255.0],  // Mage    — blue
            5 => [170.0 / 255.0, 80.0 / 255.0, 220.0 / 255.0],  // Rogue   — purple
            _ => [0.5, 0.5, 0.5],
        }
    } else if creature_type == CT_ARCHER {
        [230.0 / 255.0, 130.0 / 255.0, 40.0 / 255.0]
    } else if creature_type == CT_BRUTE {
        [150.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0]
    } else if creature_type == CT_GOBLIN {
        [95.0 / 255.0, 140.0 / 255.0, 70.0 / 255.0]
    } else {
        [0.3, 0.3, 0.3]
    }
}
