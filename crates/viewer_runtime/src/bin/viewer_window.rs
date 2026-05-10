//! `viewer_window` — Phase A.5 windowed driver for `viewer_runtime`.
//!
//! Opens a winit window, drives the Vulkan-backed `voxel_engine`
//! renderer through `Renderer::render_frame_gpu` + the swapchain's
//! `present_blit`, and ticks the sim on a 100ms fixed-step cadence
//! independent of the render rate.
//!
//! Phase A.5 deliberately renders **no scene content** — the
//! Scene-to-renderer bridge (per-tick voxel grid upload covering all
//! agents + terrain) is Phase B's slice. This binary proves the
//! windowing + present pipeline works against an empty objects list,
//! which produces the renderer's sky/sun pass with no obstructing
//! geometry. The window's title bar surfaces sim state every fixed
//! tick: `seed | tick | settlers | monsters | score`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p viewer_runtime --bin viewer_window --release [seed]
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use viewer_runtime::{ViewerApp, VoxelBridge};
use voxel_engine::app::App as _;
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::{RendererConfig, VoxelRenderer};
use voxel_engine::scene::config::SceneConfig;
use voxel_engine::scene::Scene;
use voxel_engine::ui::EguiState;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const SIM_TICK_PERIOD: Duration = Duration::from_millis(100);
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Cell count along each axis of the voxel-bridge grid. 128³ × 1B
/// per cell = 2 MB of CPU + GPU memory; ~10 µs to upload at PCIe 4.0
/// bandwidth, called once per fixed-step tick.
const VOXEL_GRID_DIM: u32 = 128;
/// World-space length the bridge grid covers along each axis.
/// Wave_defense uses ±64 around origin (settlers at radius 8,
/// spawners at radius 60); 128 covers it with 1 unit per cell.
const VOXEL_WORLD_EXTENT: f32 = 128.0;

/// Top-down observer camera. boss_fight has 6 stationary agents
/// in a ±16-unit window; 35 units up frames them tightly without
/// wasting screen real estate.
fn observer_camera() -> FreeCamera {
    FreeCamera::new(glam::Vec3::new(0.0, 35.0, 0.0), glam::Vec3::ZERO)
}

struct WindowedViewer {
    seed: u64,
    app: ViewerApp,
    scene: Scene,
    camera: FreeCamera,
    last_tick: Instant,
    /// Constructed lazily on first `resumed()` — winit 0.30 doesn't
    /// give a window until the event loop is running.
    gfx: Option<Gfx>,
}

struct Gfx {
    window: Arc<Window>,
    ctx: VulkanContext,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
    /// Phase B: world-grid bridge. `Option` because it's allocated
    /// after `app.setup()` populates the snapshot — first refresh
    /// happens before the first render so the grid isn't empty.
    bridge: VoxelBridge,
    /// Phase C: HUD/plot overlay via egui-ash. Painted on top of
    /// the swapchain image after the voxel present_blit.
    egui: EguiState,
    /// One-shot command pool for egui's texture-upload submits
    /// (font atlas, etc.). Outlives every frame.
    egui_command_pool: ash::vk::CommandPool,
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

        // egui state: framebuffers cover the swapchain image views;
        // texture-upload command pool is a one-shot transient
        // pool on the graphics queue (egui-ash-renderer submits its
        // own staging copies for font atlas uploads).
        let egui = EguiState::new(
            &ctx,
            swapchain.surface_format(),
            swapchain.image_views(),
            ash::vk::Extent2D {
                width: WINDOW_W,
                height: WINDOW_H,
            },
            &window,
        )
        .expect("EguiState::new failed");
        let gq = ctx.graphics_queue().expect("graphics queue");
        let egui_command_pool = unsafe {
            ctx.device()
                .create_command_pool(
                    &ash::vk::CommandPoolCreateInfo::default()
                        .flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
                        .queue_family_index(gq.family_index),
                    None,
                )
                .expect("egui texture-upload command pool")
        };

        // Setup primes ViewerApp's snapshot caches via
        // refresh_snapshot — needed before the first bridge.refresh
        // so the initial paint reflects post-setup state, not
        // pre-init zeros.
        self.app
            .setup(&mut self.scene)
            .expect("ViewerApp::setup failed");

        // Allocate the world-grid bridge + paint the initial frame
        // so render_frame_gpu has something to draw before the first
        // sim tick fires (otherwise the first ~100ms is empty).
        let mut bridge = VoxelBridge::new(
            &ctx,
            self.app.palette(),
            VOXEL_GRID_DIM,
            VOXEL_WORLD_EXTENT,
        )
        .expect("VoxelBridge::new failed");
        bridge
            .refresh(&ctx, &self.app)
            .expect("VoxelBridge::refresh (initial) failed");

        // Kick off the redraw loop. winit 0.30 doesn't send
        // RedrawRequested on its own past the initial expose;
        // about_to_wait below keeps it pumping.
        window.request_redraw();

        self.gfx = Some(Gfx {
            window,
            ctx,
            swapchain,
            renderer,
            bridge,
            egui,
            egui_command_pool,
        });
    }

    /// Drive continuous animation. winit calls about_to_wait whenever
    /// the event queue drains; without an explicit request_redraw
    /// here, RedrawRequested fires only on OS-driven invalidation
    /// (resize, expose) and the title bar / sim tick freeze.
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
        // Route every WindowEvent through egui first. egui returns
        // EventResponse.consumed=true for events it handled (mouse
        // clicks on egui widgets, text input into egui textfields,
        // etc.) — once we add interactive panels we'll skip our own
        // handling for consumed events. Phase C just paints; nothing
        // to skip yet.
        if let Some(gfx) = self.gfx.as_mut() {
            let _ = gfx.egui.handle_window_event(&gfx.window, &event);
        }
        match event {
            WindowEvent::CloseRequested => {
                eprintln!(
                    "[viewer_window] CloseRequested — exit at tick={}",
                    self.app.sim_tick(),
                );
                // Full cleanup before VulkanContext drops. Each
                // owned Vulkan resource has a `destroy(ctx)` method
                // (no Drop impls upstream — they all want the
                // VulkanContext, which Rust's Drop signature can't
                // pass). Order: wait_idle → bridge → swapchain →
                // renderer → drop ctx implicitly. wait_idle ensures
                // no in-flight commands are still touching these
                // resources when we destroy them.
                if let Some(mut gfx) = self.gfx.take() {
                    let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                    gfx.bridge.destroy(&gfx.ctx);
                    unsafe {
                        gfx.ctx
                            .device()
                            .destroy_command_pool(gfx.egui_command_pool, None);
                    }
                    gfx.egui.destroy(&gfx.ctx);
                    gfx.swapchain.destroy(&gfx.ctx);
                    gfx.renderer.destroy(&gfx.ctx);
                    // ctx + window drop here naturally.
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Catch up on fixed-step sim ticks. If many ticks
                // elapsed (e.g. window was hidden), bound the catch-up
                // at 4 to avoid runaway after a long pause — better to
                // skip than to freeze. Each tick refreshes the voxel
                // bridge so the rendered cells reflect the new
                // positions.
                let mut ticks_this_frame: u32 = 0;
                while self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks_this_frame < 4 {
                    self.app.tick(&mut self.scene, 0.1);
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                    if let Some(gfx) = self.gfx.as_mut() {
                        if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                            eprintln!("[viewer_window] VoxelBridge::refresh failed: {e}");
                        }
                    }
                }
                // Build the title before re-borrowing `self.gfx` mutably
                // (winit's set_title is borrow-conservative).
                let title = self.title_for_tick(self.app.sim_tick());
                let hud_state = HudState::from_app(&self.app);
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);

                    // Single render object — voxel_engine's fragment
                    // shader resolves per-cell colour through
                    // `palette_tex[voxel_id]`, so all four
                    // creature types render with their distinct
                    // palette colours from one draw.
                    let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
                    if let Err(e) = gfx
                        .renderer
                        .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                    {
                        eprintln!("[viewer_window] render_frame_gpu failed: {e}");
                        return;
                    }

                    // Run egui for the frame (panels + plots), then
                    // present_blit_with_overlay paints the result
                    // on top of the swapchain image after the voxel
                    // blit.
                    gfx.egui.run(&gfx.window, |ctx| paint_hud(ctx, &hud_state));
                    let gq = gfx.ctx.graphics_queue().expect("graphics queue");
                    let egui_pool = gfx.egui_command_pool;
                    let ctx_ref = &gfx.ctx;
                    let egui_ref = &mut gfx.egui;
                    if let Err(e) = gfx.swapchain.present_blit_with_overlay(
                        ctx_ref,
                        gfx.renderer.light_output_image(),
                        WINDOW_W,
                        WINDOW_H,
                        ash::vk::Semaphore::null(),
                        |cmd_buf, image_index| {
                            egui_ref.cmd_paint(ctx_ref, cmd_buf, image_index, gq.queue, egui_pool)
                        },
                    ) {
                        eprintln!(
                            "[viewer_window] present_blit_with_overlay failed: {e}"
                        );
                        return;
                    }
                    // about_to_wait handles the next request_redraw —
                    // no need to schedule it again here.
                }
            }
            _ => {}
        }
    }
}

impl WindowedViewer {
    fn title_for_tick(&self, tick: u64) -> String {
        let (bhp, bmax) = self.app.boss_hp();
        format!(
            "viewer_window — seed={} tick={} boss {:.0}/{:.0}  party {}  ({:.0}/{:.0})",
            self.seed,
            tick,
            bhp,
            bmax,
            self.app.party_alive_count(),
            self.app.party_total_hp(),
            self.app.party_max_total_hp(),
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    eprintln!(
        "[viewer_window] seed={seed} — opening window {WINDOW_W}x{WINDOW_H} \
         (Phase A.5: empty scene, no entities rendered yet)"
    );

    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    let app = ViewerApp::new(seed);
    let scene = Scene::new_headless(SceneConfig::default());
    let camera = observer_camera();

    let mut viewer = WindowedViewer {
        seed,
        app,
        scene,
        camera,
        last_tick: Instant::now(),
        gfx: None,
    };

    event_loop
        .run_app(&mut viewer)
        .expect("event_loop.run_app failed");
}

// Suppress unused — RendererConfig is the type Phase B will read
// from CLI args / config; pull it in now so the import doesn't churn
// the file later.
#[allow(dead_code)]
fn _phase_b_marker() -> RendererConfig {
    RendererConfig::default()
}

/// Snapshot of sim state captured per frame, handed to the egui
/// paint closure. Decoupled from `ViewerApp` so the closure doesn't
/// need to borrow the app while gfx is borrowed mutably.
struct HudState {
    tick: u64,
    boss_alive: bool,
    boss_hp: f32,
    boss_max_hp: f32,
    party: Vec<UnitHud>,
    party_total_hp: f32,
    party_max_total_hp: f32,
    party_alive: u32,
    terminated_at_tick: Option<u64>,
}

struct UnitHud {
    slot: usize,
    alive: bool,
    hp: f32,
    max_hp: f32,
    stunned: bool,
}

impl HudState {
    fn from_app(app: &ViewerApp) -> Self {
        let (bhp, bmax) = app.boss_hp();
        let mut party = Vec::with_capacity(5);
        // Party = slots 1..=5 (heroes). Slot 0 is the boss.
        let alive = app.alive();
        let hp = app.hp();
        let max = app.max_hp();
        for slot in 1..alive.len() {
            party.push(UnitHud {
                slot,
                alive: alive[slot] != 0,
                hp: hp[slot],
                max_hp: max[slot],
                stunned: app.is_stunned(slot),
            });
        }
        Self {
            tick: app.sim_tick(),
            boss_alive: app.boss_alive(),
            boss_hp: bhp,
            boss_max_hp: bmax,
            party,
            party_total_hp: app.party_total_hp(),
            party_max_total_hp: app.party_max_total_hp(),
            party_alive: app.party_alive_count(),
            terminated_at_tick: app.terminated_at_tick,
        }
    }
}

const HERO_COLOR: egui::Color32 = egui::Color32::from_rgb(60, 130, 220);
const BOSS_COLOR: egui::Color32 = egui::Color32::from_rgb(220, 60, 60);
const STUN_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 215, 80);

fn hp_bar(ui: &mut egui::Ui, label: &str, hp: f32, max_hp: f32, color: egui::Color32) {
    let frac = if max_hp > 0.0 { (hp / max_hp).clamp(0.0, 1.0) } else { 0.0 };
    ui.horizontal(|ui| {
        ui.add_sized([72.0, 16.0], egui::Label::new(label));
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(120.0, 14.0), egui::Sense::hover());
        let painter = ui.painter();
        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
        // Filled portion
        let mut fill_rect = rect;
        fill_rect.set_width(rect.width() * frac);
        painter.rect_filled(fill_rect, 2.0, color);
        // HP text on top
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.0}/{:.0}", hp, max_hp),
            egui::FontId::monospace(11.0),
            egui::Color32::WHITE,
        );
    });
}

/// Per-frame egui paint. Top-left "Boss + Party" panel.
fn paint_hud(ctx: &egui::Context, state: &HudState) {
    egui::Window::new("boss fight")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.label(format!("tick   {}", state.tick));
            ui.separator();
            ui.colored_label(BOSS_COLOR, "BOSS");
            if state.boss_alive {
                hp_bar(ui, "hp", state.boss_hp, state.boss_max_hp, BOSS_COLOR);
            } else {
                ui.colored_label(egui::Color32::DARK_GRAY, "  defeated");
            }
            ui.separator();
            ui.colored_label(
                HERO_COLOR,
                format!(
                    "PARTY  ({} alive  {:.0}/{:.0} hp)",
                    state.party_alive, state.party_total_hp, state.party_max_total_hp,
                ),
            );
            for unit in &state.party {
                let label = format!("hero {}", unit.slot);
                if !unit.alive {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [72.0, 16.0],
                            egui::Label::new(egui::RichText::new(&label).color(egui::Color32::DARK_GRAY)),
                        );
                        ui.colored_label(egui::Color32::DARK_GRAY, "  KO");
                    });
                } else {
                    let color = if unit.stunned { STUN_COLOR } else { HERO_COLOR };
                    hp_bar(ui, &label, unit.hp, unit.max_hp, color);
                    if unit.stunned {
                        ui.horizontal(|ui| {
                            ui.add_sized([72.0, 12.0], egui::Label::new(""));
                            ui.colored_label(STUN_COLOR, "  ⚡ stunned");
                        });
                    }
                }
            }
            if let Some(t) = state.terminated_at_tick {
                ui.separator();
                let label = if state.boss_alive {
                    "PARTY WIPED"
                } else {
                    "BOSS DEFEATED"
                };
                let color = if state.boss_alive { BOSS_COLOR } else { HERO_COLOR };
                ui.colored_label(color, format!("{}  (tick {})", label, t));
            }
        });
}
