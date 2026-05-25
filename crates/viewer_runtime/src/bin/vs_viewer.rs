//! `vs_viewer` — windowed viewer for `sims::vampire_survivors`.
//!
//! Opens a winit window, drives voxel_engine's Vulkan renderer, and
//! ticks the VS sim at 100ms wall-clock cadence. Top-down camera framed
//! on the arena centre (96×96 open floor, origin at voxel (48,48)).
//!
//! Usage:
//!
//! ```text
//! cargo run -p viewer_runtime --bin vs_viewer --release [SEED]
//! ```
//!
//! With no SEED, defaults to `0x5_F00D_CAFE_0001`.
//!
//! What you should see:
//!
//! - A top-down view of a flat grey 96×96 voxel arena.
//! - Cyan voxel splats for the player agent(s).
//! - Orange-red voxel splats for enemy agents.
//! - Purple voxel splats for spawner beacons.
//! - Per-tick (100ms wall clock) agents move; player dies -> auto-restart.
//!
//! NOTE: The mesh-overlay pass (glTF character models) is intentionally
//! OMITTED for vs_viewer -- the voxel splats from VsBridge are sufficient
//! to show agents and the VS roles don't map to dungeon creature-type
//! mesh slots. Visuals are the flat-voxel layer only.

use std::sync::Arc;
use std::time::{Duration, Instant};

use viewer_runtime::vs::{VsViewerApp, VsBridge, VsRole};
use viewer_runtime::vs_ui;
use viewer_runtime::{BRIDGE_DIM_X, BRIDGE_DIM_Z};
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Wall-clock period between sim ticks.
const SIM_TICK_PERIOD: Duration = Duration::from_millis(50);
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Default seed.
const DEFAULT_SEED: u64 = 0x5_F00D_CAFE_0001;

/// Top-down observer camera framed on the VS arena centre.
/// The arena is 96x96 cells centred at voxel (48, 48).
fn observer_camera(bridge_dim_x: u32, bridge_dim_z: u32) -> FreeCamera {
    let cx = bridge_dim_x as f32 / 2.0;
    let cz = bridge_dim_z as f32 / 2.0;
    let height = bridge_dim_x.max(bridge_dim_z) as f32 + 8.0;
    // Top-down isometric: eye above + equal diagonal X/Z offset (45deg azimuth).
    FreeCamera::new(
        glam::Vec3::new(cx + height * 0.5, height, cz + height * 0.5),
        glam::Vec3::new(cx, 0.0, cz),
    )
}

struct WindowedVsViewer {
    seed: u64,
    app: VsViewerApp,
    camera: FreeCamera,
    last_tick: Instant,
    /// Constructed lazily on first `resumed()`.
    gfx: Option<Gfx>,
    /// Wall-clock instant at which the current run terminated.
    terminated_at_wall: Option<Instant>,
    /// Pause flag -- toggled by Space.
    paused: bool,
    /// Camera pan offset in renderer XZ.
    pan_xz: glam::Vec2,
    /// Camera zoom factor.
    zoom: f32,
    /// Held-key set for per-frame pan/zoom integration.
    held_keys: std::collections::HashSet<String>,
    /// Total runs in this session.
    session_runs: u32,
    /// Total ticks accumulated across runs.
    session_tick_total: u64,
    /// Host-side game-UI state.
    progress: vs_ui::PlayerProgress,
    /// HUD + modal-screen declaration (rebuilt per run; menus injected live).
    ui_model: engine_ui::UiModel,
    /// Per-frame HUD values (rebuilt each frame from sim readback).
    ui_data: engine_ui::UiData,
    /// Name of the active modal screen, if any ("level_up" / "dead"). While
    /// `Some`, the sim is paused and `engine_ui` renders that screen.
    active_screen: Option<String>,
}

struct Gfx {
    window: Arc<Window>,
    ctx: VulkanContext,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
    bridge: VsBridge,
    /// egui overlay state (HUD / menu / death screens).
    egui: voxel_engine::ui::EguiState,
    /// Command pool for egui's texture-staging submits (cmd_paint needs one;
    /// VulkanContext exposes no accessor, so we own one here). Graphics family.
    egui_cmd_pool: ash::vk::CommandPool,
    /// Graphics queue used for egui texture uploads.
    egui_queue: ash::vk::Queue,
}

impl ApplicationHandler for WindowedVsViewer {
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
            VsBridge::new(&ctx).expect("VsBridge::new failed");
        bridge
            .refresh(&ctx, &self.app)
            .expect("VsBridge::refresh (initial) failed");

        // egui overlay: matches the swapchain's format/views/extent.
        let egui = voxel_engine::ui::EguiState::new(
            &ctx,
            swapchain.surface_format(),
            swapchain.image_views(),
            swapchain.extent(),
            &window,
        )
        .expect("EguiState::new failed");
        // egui's cmd_paint needs a graphics queue + a command pool for its
        // texture-staging submits. VulkanContext exposes no command-pool
        // accessor, so we own one on the graphics family.
        let gq = ctx
            .graphics_queue()
            .expect("graphics queue required for egui overlay");
        let pool_ci = ash::vk::CommandPoolCreateInfo::default()
            .queue_family_index(gq.family_index)
            .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let egui_cmd_pool = unsafe { ctx.device().create_command_pool(&pool_ci, None) }
            .expect("egui command pool");

        window.request_redraw();

        self.gfx = Some(Gfx {
            window,
            ctx,
            swapchain,
            renderer,
            bridge,
            egui,
            egui_cmd_pool,
            egui_queue: gq.queue,
        });
    }

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
        // Route the event into egui first. If egui consumed it (e.g. a click
        // on a menu card or a hovered window), don't also process it here.
        if let Some(gfx) = self.gfx.as_mut() {
            let resp = gfx.egui.handle_window_event(&gfx.window, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent { ref logical_key, state, repeat: false, .. },
                ..
            } => {
                let key_name = key_to_name(logical_key);
                match state {
                    ElementState::Pressed => {
                        if let Some(n) = key_name { self.held_keys.insert(n); }
                    }
                    ElementState::Released => {
                        if let Some(n) = key_name { self.held_keys.remove(&n); }
                        return;
                    }
                }
                match logical_key {
                    Key::Named(NamedKey::Space) => {
                        self.paused = !self.paused;
                        eprintln!(
                            "[vs_viewer] {} @ tick {}",
                            if self.paused { "PAUSED" } else { "RESUMED" },
                            self.app.sim_tick(),
                        );
                    }
                    Key::Character(c) if c.eq_ignore_ascii_case("r") => {
                        self.restart_run();
                    }
                    _ => {}
                }
            }
            WindowEvent::CloseRequested => {
                eprintln!(
                    "[vs_viewer] CloseRequested -- exit at tick={}",
                    self.app.sim_tick(),
                );
                if self.session_runs > 0 {
                    let mean_ticks = self.session_tick_total / self.session_runs as u64;
                    eprintln!(
                        "[vs_viewer] final session: {} runs, mean {} ticks",
                        self.session_runs,
                        mean_ticks,
                    );
                }
                if let Some(mut gfx) = self.gfx.take() {
                    let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                    unsafe {
                        gfx.ctx
                            .device()
                            .destroy_command_pool(gfx.egui_cmd_pool, None);
                    }
                    gfx.egui.destroy(&gfx.ctx);
                    gfx.bridge.destroy(&gfx.ctx);
                    gfx.swapchain.destroy(&gfx.ctx);
                    gfx.renderer.destroy(&gfx.ctx);
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // The sim is frozen while a modal screen (level-up / death) is
                // up; treat that like a pause for stepping purposes.
                let frozen = self.paused || self.active_screen.is_some();

                // Push the player's intent + current upgrade levels into the
                // sim's runtime config (`config.ctl.*`) before this frame's
                // ticks. WASD → normalized move vector; progress → weapons.
                {
                    let (mut mx, mut my) = (0.0f32, 0.0f32);
                    if self.held_keys.contains("w") { my += 1.0; }
                    if self.held_keys.contains("s") { my -= 1.0; }
                    if self.held_keys.contains("d") { mx += 1.0; }
                    if self.held_keys.contains("a") { mx -= 1.0; }
                    // Freeze movement while a modal is up.
                    if frozen { mx = 0.0; my = 0.0; }
                    let len = (mx * mx + my * my).sqrt();
                    if len > 1e-3 { mx /= len; my /= len; }
                    self.app.state.set_config_ctl_move_x(mx);
                    self.app.state.set_config_ctl_move_y(my);
                    self.app.state.set_config_ctl_bolt_level(self.progress.bolt_level);
                    self.app.state.set_config_ctl_bolt_rate_level(self.progress.bolt_rate_level);
                    self.app.state.set_config_ctl_nova_level(self.progress.nova_level);
                    self.app.state.set_config_ctl_move_level(self.progress.move_level);
                    self.app.state.set_config_ctl_garlic_level(self.progress.garlic_level);
                    self.app.state.set_config_ctl_whip_level(self.progress.whip_level);
                }

                // Catch up on fixed-step sim ticks. Bound at 8 to avoid runaway.
                if frozen {
                    self.last_tick = Instant::now();
                }
                let mut ticks_this_frame: u32 = 0;
                while !frozen
                    && self.last_tick.elapsed() >= SIM_TICK_PERIOD
                    && ticks_this_frame < 8
                {
                    self.app.step();
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                    if let Some(gfx) = self.gfx.as_mut() {
                        if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                            eprintln!("[vs_viewer] VsBridge::refresh failed: {e}");
                        }
                    }
                }

                // Read the player's XP from the materialized view and drive the
                // level-up / death state machine.
                let xp = self.app.player_xp();
                self.progress.kills = xp as u32;
                if self.active_screen.is_none()
                    && self.app.alive()
                    && self.progress.check_level_up(xp)
                {
                    // Open a fresh seeded 3-card menu and pause the run.
                    self.ui_model.screens =
                        vec![vs_ui::menu_screen(self.app.seed, self.progress.last_level)];
                    self.active_screen = Some("level_up".into());
                    eprintln!(
                        "[vs_viewer] level up -> {} (xp={xp:.0}) — menu open",
                        self.progress.last_level,
                    );
                }
                // Death: show the summary screen instead of (or before) the
                // auto-restart hold.
                if self.active_screen.is_none() && !self.app.alive() {
                    self.ui_model.screens = vec![vs_ui::death_screen()];
                    self.active_screen = Some("dead".into());
                }

                // Build this frame's HUD values from the snapshot.
                let hp = self.app.player_hp();
                let enemies = self.app.enemy_count();
                let tick = self.app.sim_tick();
                vs_ui::build_data(
                    &mut self.ui_data,
                    hp,
                    vs_ui::PLAYER_HP_MAX,
                    xp,
                    self.progress.kills,
                    tick,
                    enemies,
                );

                // Auto-restart when the player is dead (terminated_at_tick is Some).
                const POST_TERMINATION_HOLD: Duration = Duration::from_secs(3);
                if !self.paused
                    && self.app.terminated_at_tick.is_some()
                    && self.terminated_at_wall.is_none()
                {
                    self.terminated_at_wall = Some(Instant::now());
                    eprintln!(
                        "[vs_viewer] player dead at tick {}",
                        self.app.sim_tick(),
                    );
                }
                // Fallback auto-restart: if the player leaves the death screen
                // up (or it never opened), re-seed after the hold so the demo
                // reel keeps cycling. The death-screen Restart button does the
                // same thing immediately via `restart_run`.
                if let Some(t) = self.terminated_at_wall {
                    if !self.paused && t.elapsed() >= POST_TERMINATION_HOLD {
                        self.session_runs += 1;
                        self.session_tick_total += self.app.sim_tick();
                        let mean_ticks = self.session_tick_total / self.session_runs as u64;
                        eprintln!(
                            "[vs_viewer] session: {} runs, mean {} ticks",
                            self.session_runs, mean_ticks,
                        );
                        self.restart_run();
                    }
                }

                // Integrate held-key camera pan/zoom. WASD now drives the
                // player (see the movement push before the step loop); the
                // arrow keys retain camera pan.
                {
                    const PAN_SPEED: f32 = 0.1;
                    const ZOOM_RATE: f32 = 0.99;
                    let mut dx = 0.0_f32;
                    let mut dz = 0.0_f32;
                    if self.held_keys.contains("ArrowUp") { dz -= PAN_SPEED; }
                    if self.held_keys.contains("ArrowDown") { dz += PAN_SPEED; }
                    if self.held_keys.contains("ArrowLeft") { dx -= PAN_SPEED; }
                    if self.held_keys.contains("ArrowRight") { dx += PAN_SPEED; }
                    self.pan_xz.x += dx;
                    self.pan_xz.y += dz;
                    if self.held_keys.contains("=") || self.held_keys.contains("+") {
                        self.zoom *= ZOOM_RATE;
                    }
                    if self.held_keys.contains("-") || self.held_keys.contains("_") {
                        self.zoom /= ZOOM_RATE;
                    }
                    self.zoom = self.zoom.clamp(0.15, 3.0);

                    // Follow the player: its world pos maps to a voxel cell via the
                    // same +DIM/2 offset the bridge uses, so the camera centers on it.
                    let (pcx, pcz) = self
                        .app
                        .agents()
                        .iter()
                        .find(|a| a.role == VsRole::Player)
                        .map(|pl| (pl.pos[0], pl.pos[1]))
                        .unwrap_or((0.0, 0.0));
                    let cx = BRIDGE_DIM_X as f32 / 2.0 + pcx + self.pan_xz.x;
                    let cz = BRIDGE_DIM_Z as f32 / 2.0 + pcz + self.pan_xz.y;
                    let base_height = BRIDGE_DIM_X.max(BRIDGE_DIM_Z) as f32 + 8.0;
                    let height = base_height * self.zoom;
                    // Top-down isometric, recentred on the player each frame.
                    self.camera = FreeCamera::new(
                        glam::Vec3::new(cx + height * 0.5, height, cz + height * 0.5),
                        glam::Vec3::new(cx, 0.0, cz),
                    );
                }

                let title = self.title_for_tick(self.app.sim_tick());
                // The UI action the user triggered this frame (card click /
                // restart button), captured out of the egui closure and applied
                // after the `gfx` borrow ends.
                let mut ui_action: Option<engine_ui::UiAction> = None;
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);
                    let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
                    if let Err(e) = gfx
                        .renderer
                        .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                    {
                        eprintln!("[vs_viewer] render_frame_gpu failed: {e}");
                        return;
                    }
                    // egui overlay: run the UI for this frame, then paint it on
                    // top of the voxel render inside present_blit's overlay pass.
                    let model = &self.ui_model;
                    let data = &self.ui_data;
                    let active = self.active_screen.as_deref();
                    gfx.egui.run(&gfx.window, |ectx| {
                        ui_action = engine_ui::draw(ectx, model, data, active);
                    });
                    // Capture the fields the FnOnce overlay closure needs so the
                    // borrow of `gfx.swapchain` doesn't conflict with `gfx.egui`.
                    let Gfx {
                        ref ctx,
                        ref mut swapchain,
                        ref renderer,
                        ref mut egui,
                        egui_cmd_pool,
                        egui_queue,
                        ..
                    } = *gfx;
                    let present_result = swapchain.present_blit_with_overlay(
                        ctx,
                        renderer.light_output_image(),
                        WINDOW_W,
                        WINDOW_H,
                        ash::vk::Semaphore::null(),
                        |cmd, image_index| {
                            egui.cmd_paint(ctx, cmd, image_index, egui_queue, egui_cmd_pool)
                        },
                    );
                    if let Err(e) = present_result {
                        eprintln!("[vs_viewer] present_blit_with_overlay failed: {e}");
                    }
                }

                // Apply the UI action (outside the `gfx` borrow). A menu card
                // raises an upgrade level and closes the menu (unpausing); the
                // death-screen Restart re-seeds the run.
                match ui_action {
                    Some(action @ engine_ui::UiAction::Increment(_)) => {
                        self.progress.apply(&action);
                        self.active_screen = None;
                        self.ui_model.screens.clear();
                    }
                    Some(engine_ui::UiAction::Restart) => {
                        self.restart_run();
                    }
                    None => {}
                }
            }
            _ => {}
        }
    }
}

impl WindowedVsViewer {
    /// Re-seed into a fresh run: new sim app, reset progress + UI screens,
    /// rebuild the render bridge. Shared by the R key, the death-screen
    /// Restart button, and the auto-restart-on-death hold.
    fn restart_run(&mut self) {
        let next_seed = self.seed.wrapping_add(1);
        eprintln!(
            "[vs_viewer] restart: seed 0x{:X} -> 0x{:X}",
            self.seed, next_seed,
        );
        let Some(new_app) = VsViewerApp::try_new(next_seed) else {
            eprintln!("[vs_viewer] restart failed -- holding on current run.");
            self.terminated_at_wall = Some(Instant::now());
            return;
        };
        self.app = new_app;
        self.seed = next_seed;
        self.last_tick = Instant::now();
        self.terminated_at_wall = None;
        self.paused = false;
        self.pan_xz = glam::Vec2::ZERO;
        self.zoom = 1.0;
        self.progress = vs_ui::PlayerProgress::default();
        self.active_screen = None;
        self.ui_model = vs_ui::hud_model();
        self.ui_data = engine_ui::UiData::new();
        if let Some(gfx) = self.gfx.as_mut() {
            let _ = unsafe { gfx.ctx.device().device_wait_idle() };
            let old_bridge = std::mem::replace(
                &mut gfx.bridge,
                VsBridge::new(&gfx.ctx).expect("VsBridge::new (restart) failed"),
            );
            old_bridge.destroy(&gfx.ctx);
            if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                eprintln!("[vs_viewer] post-restart refresh failed: {e}");
            }
        }
    }

    fn title_for_tick(&self, tick: u64) -> String {
        let agents = self.app.agents();
        let players = agents.iter().filter(|a| a.role == VsRole::Player).count();
        let enemies = agents.iter().filter(|a| a.role == VsRole::Enemy).count();
        format!(
            "vs_viewer -- seed=0x{:X} tick={} alive={} (player={} enemy={})",
            self.seed, tick, agents.len(), players, enemies,
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
        "[vs_viewer] seed=0x{seed:X} -- opening window {WINDOW_W}x{WINDOW_H}"
    );
    eprintln!(
        "[vs_viewer] What you should see: top-down view of a flat grey 96x96 \
         voxel arena. Cyan = player, orange-red = enemies, purple = spawners. \
         Tick advances every 100ms; player death triggers auto-restart."
    );

    let app = match VsViewerApp::try_new(seed) {
        Some(a) => a,
        None => {
            eprintln!(
                "[vs_viewer] no compatible wgpu adapter found for sim runtime -- \
                 cannot initialize. Try running on a host with a discrete GPU \
                 or software fallback (lavapipe, etc.)."
            );
            std::process::exit(2);
        }
    };
    let camera = observer_camera(BRIDGE_DIM_X, BRIDGE_DIM_Z);

    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut viewer = WindowedVsViewer {
        seed,
        app,
        camera,
        last_tick: Instant::now(),
        gfx: None,
        terminated_at_wall: None,
        paused: false,
        pan_xz: glam::Vec2::ZERO,
        zoom: 1.0,
        held_keys: std::collections::HashSet::new(),
        session_runs: 0,
        session_tick_total: 0,
        progress: vs_ui::PlayerProgress::default(),
        ui_model: vs_ui::hud_model(),
        ui_data: engine_ui::UiData::new(),
        active_screen: None,
    };
    event_loop
        .run_app(&mut viewer)
        .expect("event_loop.run_app failed");
}

/// Map a winit logical_key to a stable string name for `held_keys`.
fn key_to_name(k: &Key) -> Option<String> {
    match k {
        Key::Character(c) => {
            let lower = c.to_lowercase();
            match lower.as_str() {
                "w" | "a" | "s" | "d" | "=" | "-" | "+" | "_" => Some(lower),
                _ => None,
            }
        }
        Key::Named(NamedKey::ArrowUp)    => Some("ArrowUp".into()),
        Key::Named(NamedKey::ArrowDown)  => Some("ArrowDown".into()),
        Key::Named(NamedKey::ArrowLeft)  => Some("ArrowLeft".into()),
        Key::Named(NamedKey::ArrowRight) => Some("ArrowRight".into()),
        _ => None,
    }
}
