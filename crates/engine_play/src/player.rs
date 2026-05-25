//! The generic player loop. The per-frame logic — controls resolve →
//! `set_input` → `step` → snapshot → `UiData` build → `engine_ui::draw` —
//! lives in [`update`], which is **separable from the winit/GPU shell** so it
//! can be driven headlessly in tests. The windowed [`Player`]
//! (`winit::ApplicationHandler`) wraps `update` with a Vulkan/egui surface.
//!
//! Host-side state (level/menu/death) is generalized from
//! `viewer_runtime::vs_ui`: the player's "level" is read from a materialized
//! view named by [`PlayerConfig::level_view`] at [`PlayerConfig::player_slot`];
//! crossing an integer level opens the `level_up` modal; a dead followed agent
//! opens the `dead` modal. `UiAction`s are applied via host counters mirrored
//! into `set_input` (e.g. `Increment("bolt_level")` → `set_config_ctl_bolt_level`
//! is what a real runtime exposes; here we write `ctl.bolt_level`).

use std::collections::{HashMap, HashSet};

use engine_play_api::{ControlsDescriptor, PlayableRuntime, RenderDescriptor};
use engine_ui::{UiAction, UiData, UiModel};

use crate::bridge::EngineBridge;
use crate::input::ControlsMapper;

/// Static configuration for the host-side level/menu/death machine. A real
/// runtime would derive these from its UI/render descriptors; for Wave 1 they
/// have sensible defaults keyed to the sample fixture.
#[derive(Clone, Debug)]
pub struct PlayerConfig {
    /// Materialized-view name holding the player's cumulative progress (XP).
    pub level_view: String,
    /// Agent slot the player occupies (for `view_value`).
    pub player_slot: u32,
    /// XP per level (crossing an integer multiple opens the menu).
    pub xp_per_level: f32,
    /// Followed-agent HP denominator for the HUD bar.
    pub player_hp_max: f32,
    /// `@runtime` field prefix that host upgrade counters write to (e.g.
    /// `"ctl"` → `Increment("bolt_level")` writes `ctl.bolt_level`).
    pub upgrade_block: String,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self {
            level_view: "xp".into(),
            player_slot: 0,
            xp_per_level: 5.0,
            player_hp_max: 100.0,
            upgrade_block: "ctl".into(),
        }
    }
}

/// Host-side run state: upgrade counters, last-seen level (for level-up edge
/// detection), the active modal screen, and the UI model/data. Generic — no
/// per-game fields; upgrades are an open `HashMap<String, f32>`.
pub struct HostState {
    pub cfg: PlayerConfig,
    /// Upgrade levels keyed by counter name (e.g. "bolt_level").
    pub upgrades: HashMap<String, f32>,
    /// Last integer level observed (level-up edge tracking).
    pub last_level: u32,
    /// Active modal screen name ("level_up" / "dead"), if any. While `Some`,
    /// the sim is frozen.
    pub active_screen: Option<String>,
    /// HUD + modal-screen declaration.
    pub ui_model: UiModel,
    /// Per-frame HUD values.
    pub ui_data: UiData,
}

impl HostState {
    pub fn new(cfg: PlayerConfig, ui_model: UiModel) -> Self {
        Self {
            cfg,
            upgrades: HashMap::new(),
            last_level: 0,
            active_screen: None,
            ui_model,
            ui_data: UiData::new(),
        }
    }

    /// Discrete level from cumulative XP.
    fn level(&self, xp: f32) -> u32 {
        (xp / self.cfg.xp_per_level).floor() as u32
    }

    /// Apply a `UiAction`. `Increment` raises a host counter; `Restart` is
    /// signalled to the caller (returns true), which re-seeds the run.
    pub fn apply_action(&mut self, action: &UiAction) -> bool {
        match action {
            UiAction::Increment(k) => {
                *self.upgrades.entry(k.clone()).or_insert(0.0) += 1.0;
                self.active_screen = None;
                false
            }
            UiAction::Restart => true,
        }
    }
}

/// What [`update`] produced this frame (for the shell + tests to act on).
#[derive(Default, Debug)]
pub struct UpdateOutput {
    /// `(field, value)` writes pushed to the runtime this frame (controls +
    /// mirrored upgrade counters), for assertions.
    pub inputs: Vec<(String, f32)>,
    /// The UI action the user triggered (card click / restart), if any.
    pub action: Option<UiAction>,
    /// True if the run is frozen (a modal is up) this frame.
    pub frozen: bool,
    /// The followed agent's world position (camera target), if any.
    pub follow_pos: Option<[f32; 3]>,
}

/// Drive one frame of the player against `rt`, painting into `bridge`'s CPU
/// grid and running the UI in the provided headless-or-windowed `egui::Context`.
///
/// Steps (mirrors `vs_viewer`'s `RedrawRequested`, minus the GPU present):
/// 1. Resolve held keys + controls descriptor → input writes; freeze movement
///    while a modal is up; push to `rt.set_input`.
/// 2. Mirror host upgrade counters into `rt.set_input` (`<block>.<counter>`).
/// 3. If not frozen, `rt.step()` then paint the snapshot into the bridge grid.
/// 4. Read XP from the level view; open `level_up` on a level edge, `dead` when
///    the followed agent is gone; build HUD `UiData`.
/// 5. Run `engine_ui::draw` in `ectx`, capturing any `UiAction`.
///
/// `paint_grid` is the CPU `PaintGrid` to paint into (the headless test passes a
/// `Painted`; the windowed shell paints via `bridge.refresh` on the GPU grid
/// and passes a throwaway sink here — see `Player`).
#[allow(clippy::too_many_arguments)]
pub fn update(
    rt: &mut dyn PlayableRuntime,
    bridge: &EngineBridge,
    host: &mut HostState,
    controls: &ControlsDescriptor,
    held: &HashSet<String>,
    pressed: &HashSet<String>,
    ectx: &egui::Context,
    paint_grid: &mut dyn crate::bridge::PaintGrid,
    last_cells: &mut Vec<(u32, u32, u32)>,
) -> UpdateOutput {
    let mut out = UpdateOutput::default();
    let frozen = host.active_screen.is_some();
    out.frozen = frozen;

    // 1. Controls → input. Movement is frozen while a modal is up.
    let mut writes = ControlsMapper::resolve(controls, held);
    if frozen {
        for (_, v) in writes.iter_mut() {
            *v = 0.0;
        }
    } else {
        // Press-mode edges fire only when not frozen.
        writes.extend(ControlsMapper::resolve_pressed(controls, pressed));
    }
    for (field, value) in &writes {
        rt.set_input(field, *value);
        out.inputs.push((field.clone(), *value));
    }

    // 2. Mirror host upgrade counters into the runtime config.
    for (counter, level) in &host.upgrades {
        let field = format!("{}.{}", host.cfg.upgrade_block, counter);
        rt.set_input(&field, *level);
        out.inputs.push((field, *level));
    }

    // 3. Step + paint (skipped while frozen).
    if !frozen {
        rt.step();
    }
    let tick = rt.tick();
    let agents = rt.agent_snapshot();
    let painted = bridge.paint(paint_grid, &agents, tick, last_cells);
    *last_cells = painted;
    out.follow_pos = bridge.followed(&agents).map(|a| a.pos);

    // 4. Level / death machine + HUD data.
    let xp = rt.view_value(&host.cfg.level_view, host.cfg.player_slot);
    let lv = host.level(xp);
    let player_alive = bridge.followed(&agents).is_some();
    if host.active_screen.is_none() {
        if player_alive && lv > host.last_level {
            host.active_screen = Some("level_up".into());
        } else if !player_alive {
            host.active_screen = Some("dead".into());
        }
    }
    host.last_level = lv;

    let hp = bridge.followed(&agents).map(|a| a.hp).unwrap_or(0.0);
    let enemies = agents.len();
    host.ui_data
        .set("hp", hp)
        .set("hp_max", host.cfg.player_hp_max)
        .set("level", lv as f32)
        .set("xp", xp)
        .set("xp_into", xp % host.cfg.xp_per_level)
        .set("xp_per_level", host.cfg.xp_per_level)
        .set("kills", xp)
        .set("time", tick as f32 * 0.1)
        .set("enemies", enemies as f32);

    // 5. Draw the UI in the provided context, capturing the action.
    let model = &host.ui_model;
    let data = &host.ui_data;
    let active = host.active_screen.clone();
    let mut action = None;
    let _ = ectx.run(egui::RawInput::default(), |ctx| {
        action = engine_ui::draw(ctx, model, data, active.as_deref());
    });
    out.action = action;
    out
}

/// A default UI model for the mock/sample game (HP/XP bars + a status line, a
/// level-up menu, a death screen). Built directly in Rust; `UiModel::from_json`
/// lands in a parallel track.
pub fn mock_ui_model() -> UiModel {
    use engine_ui::{Card, NamedScreen, Screen, Widget};
    UiModel {
        hud: vec![
            Widget::Bar {
                label: "HP".into(),
                value: "hp".into(),
                max: "hp_max".into(),
                color: [220, 40, 40],
            },
            Widget::Bar {
                label: "XP".into(),
                value: "xp_into".into(),
                max: "xp_per_level".into(),
                color: [40, 160, 240],
            },
            Widget::Text {
                template: "Lv {level}   Kills {kills}   {time}s   Enemies {enemies}".into(),
            },
        ],
        screens: vec![
            NamedScreen {
                name: "level_up".into(),
                screen: Screen::Menu {
                    title: "Level Up!".into(),
                    cards: vec![
                        Card {
                            label: "Bolt +".into(),
                            action: UiAction::Increment("bolt_level".into()),
                        },
                        Card {
                            label: "Nova +".into(),
                            action: UiAction::Increment("nova_level".into()),
                        },
                    ],
                },
            },
            NamedScreen {
                name: "dead".into(),
                screen: Screen::End {
                    title: "You Died".into(),
                    summary: vec![
                        ("Time".into(), "time".into()),
                        ("Level".into(), "level".into()),
                        ("Kills".into(), "kills".into()),
                    ],
                    restart_label: "Restart (R)".into(),
                },
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Windowed shell. Compiled always (the binary links it), but the windowed
// `run` requires a display; tests drive `update` headlessly instead.
// ---------------------------------------------------------------------------

pub use shell::Player;

mod shell {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use voxel_engine::camera::FreeCamera;
    use voxel_engine::render::VoxelRenderer;
    use voxel_engine::vulkan::instance::VulkanContext;
    use voxel_engine::vulkan::swapchain::SwapchainContext;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, KeyEvent, WindowEvent};
    use winit::event_loop::ActiveEventLoop;
    use winit::keyboard::{Key, NamedKey};
    use winit::window::{Window, WindowId};

    const WINDOW_W: u32 = 1280;
    const WINDOW_H: u32 = 720;
    const SIM_TICK_PERIOD: Duration = Duration::from_millis(50);

    /// The windowed generic player. Owns a `Box<dyn PlayableRuntime>`, the
    /// `EngineBridge`, the parsed descriptors, host state, and (lazily) the
    /// Vulkan/egui surface. The per-frame logic delegates to [`super::update`].
    pub struct Player {
        rt: Box<dyn PlayableRuntime>,
        render: RenderDescriptor,
        controls: ControlsDescriptor,
        host: HostState,
        held_keys: HashSet<String>,
        pressed_this_frame: HashSet<String>,
        last_tick: Instant,
        camera: FreeCamera,
        gfx: Option<Gfx>,
        /// CPU grid sink for `update`'s paint (the GPU grid is painted inside
        /// `bridge.refresh`; this throwaway keeps the shared paint signature).
        scratch_last: Vec<(u32, u32, u32)>,
    }

    struct Gfx {
        window: Arc<Window>,
        ctx: VulkanContext,
        swapchain: SwapchainContext,
        renderer: VoxelRenderer,
        bridge: EngineBridge,
        egui: voxel_engine::ui::EguiState,
        egui_cmd_pool: ash::vk::CommandPool,
        egui_queue: ash::vk::Queue,
    }

    impl Player {
        /// Build a player over `rt`, parsing its descriptors. The UI model is
        /// supplied directly (Wave 1; `UiModel::from_json` is a parallel track).
        pub fn new(
            rt: Box<dyn PlayableRuntime>,
            cfg: PlayerConfig,
            ui_model: UiModel,
        ) -> anyhow::Result<Self> {
            let render = RenderDescriptor::from_json(rt.render_descriptor())?;
            let controls = ControlsDescriptor::from_json(rt.controls_descriptor())?;
            let geom = crate::bridge::Geometry::from_radius(render.arena_radius);
            let camera = observer_camera(geom.dim, geom.dim);
            Ok(Self {
                rt,
                render,
                controls,
                host: HostState::new(cfg, ui_model),
                held_keys: HashSet::new(),
                pressed_this_frame: HashSet::new(),
                last_tick: Instant::now(),
                camera,
                gfx: None,
                scratch_last: Vec::new(),
            })
        }

        /// Run the windowed loop (requires a display). Returns when the window
        /// closes. Headless callers should drive [`super::update`] directly.
        pub fn run(mut self) -> anyhow::Result<()> {
            use winit::event_loop::EventLoop;
            let event_loop = EventLoop::new()?;
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            event_loop.run_app(&mut self)?;
            Ok(())
        }
    }

    /// Top-down isometric observer camera framing a `dim×dim` arena.
    fn observer_camera(dim_x: u32, dim_z: u32) -> FreeCamera {
        let cx = dim_x as f32 / 2.0;
        let cz = dim_z as f32 / 2.0;
        let height = dim_x.max(dim_z) as f32 + 8.0;
        FreeCamera::new(
            glam::Vec3::new(cx + height * 0.5, height, cz + height * 0.5),
            glam::Vec3::new(cx, 0.0, cz),
        )
    }

    impl ApplicationHandler for Player {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.gfx.is_some() {
                return;
            }
            let attrs = Window::default_attributes()
                .with_title("engine_play")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_W, WINDOW_H));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("create_window failed"),
            );
            let ctx = VulkanContext::new_with_surface_extensions(&window)
                .expect("VulkanContext::new_with_surface_extensions failed");
            let swapchain =
                SwapchainContext::new(&ctx, &window).expect("SwapchainContext::new failed");
            let renderer =
                VoxelRenderer::new(&ctx, WINDOW_W, WINDOW_H).expect("VoxelRenderer::new failed");
            let mut bridge = EngineBridge::new(&ctx, self.render.clone())
                .expect("EngineBridge::new failed");
            let agents = self.rt.agent_snapshot();
            let tick = self.rt.tick();
            bridge
                .refresh(&ctx, &agents, tick)
                .expect("EngineBridge::refresh (initial) failed");

            let egui = voxel_engine::ui::EguiState::new(
                &ctx,
                swapchain.surface_format(),
                swapchain.image_views(),
                swapchain.extent(),
                &window,
            )
            .expect("EguiState::new failed");
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
            if let Some(gfx) = self.gfx.as_mut() {
                let resp = gfx.egui.handle_window_event(&gfx.window, &event);
                if resp.consumed {
                    return;
                }
            }
            match event {
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            ref logical_key,
                            state,
                            repeat: false,
                            ..
                        },
                    ..
                } => {
                    if let Some(name) = key_to_name(logical_key) {
                        match state {
                            ElementState::Pressed => {
                                self.held_keys.insert(name.clone());
                                self.pressed_this_frame.insert(name);
                            }
                            ElementState::Released => {
                                self.held_keys.remove(&name);
                            }
                        }
                    }
                }
                WindowEvent::CloseRequested => {
                    if let Some(mut gfx) = self.gfx.take() {
                        let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                        unsafe {
                            gfx.ctx.device().destroy_command_pool(gfx.egui_cmd_pool, None);
                        }
                        gfx.egui.destroy(&gfx.ctx);
                        gfx.bridge.destroy(&gfx.ctx);
                        gfx.swapchain.destroy(&gfx.ctx);
                        gfx.renderer.destroy(&gfx.ctx);
                    }
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.redraw();
                }
                _ => {}
            }
        }
    }

    impl Player {
        /// One windowed frame: catch up sim ticks via [`super::update`] (GPU
        /// paint through `bridge.refresh`), then present with the egui overlay.
        fn redraw(&mut self) {
            let frozen = self.host.active_screen.is_some();
            if frozen {
                self.last_tick = Instant::now();
            }

            // Pull the GPU bridge out so we can call `update` with `&self`
            // borrows + paint the GPU grid. We use a throwaway sink for the
            // shared `update` paint, then re-run the GPU paint via refresh.
            let mut ticks = 0u32;
            while !frozen && self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks < 8 {
                self.last_tick += SIM_TICK_PERIOD;
                ticks += 1;
            }

            // Run the per-frame logic once (controls/step/snapshot/UI). We need
            // the bridge for `paint`; the windowed path paints the GPU grid via
            // `refresh`, so feed `update` a throwaway sink and let `refresh`
            // own the real grid.
            let held = self.held_keys.clone();
            let pressed = std::mem::take(&mut self.pressed_this_frame);

            // The egui context lives in EguiState; we run `update`'s logic
            // against a fresh headless context for the action, then re-run the
            // *visible* egui inside EguiState::run below for the actual paint.
            // To avoid double-stepping, do the step/snapshot here via update
            // with a scratch grid, then GPU-refresh from the same snapshot.
            let mut scratch = crate::bridge::Painted::new();
            let out = {
                let Some(gfx) = self.gfx.as_ref() else { return };
                let ectx = egui::Context::default();
                super::update(
                    self.rt.as_mut(),
                    &gfx.bridge,
                    &mut self.host,
                    &self.controls,
                    &held,
                    &pressed,
                    &ectx,
                    &mut scratch,
                    &mut self.scratch_last,
                )
            };

            // GPU paint from the post-step snapshot.
            let agents = self.rt.agent_snapshot();
            let tick = self.rt.tick();
            if let Some(gfx) = self.gfx.as_mut() {
                if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &agents, tick) {
                    eprintln!("[engine_play] bridge.refresh failed: {e}");
                }
            }

            // Camera follows the followed agent.
            if let Some(pos) = out.follow_pos {
                let geom = self
                    .gfx
                    .as_ref()
                    .map(|g| g.bridge.geometry())
                    .unwrap_or_else(|| crate::bridge::Geometry::from_radius(self.render.arena_radius));
                let cx = geom.dim as f32 / 2.0 + pos[0];
                let cz = geom.dim as f32 / 2.0 + pos[1];
                let height = geom.dim as f32 + 8.0;
                self.camera = FreeCamera::new(
                    glam::Vec3::new(cx + height * 0.5, height, cz + height * 0.5),
                    glam::Vec3::new(cx, 0.0, cz),
                );
            }

            // Present with the egui overlay.
            let model = &self.host.ui_model;
            let data = &self.host.ui_data;
            let active = self.host.active_screen.clone();
            let mut ui_action: Option<UiAction> = None;
            if let Some(gfx) = self.gfx.as_mut() {
                let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
                if let Err(e) = gfx
                    .renderer
                    .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                {
                    eprintln!("[engine_play] render_frame_gpu failed: {e}");
                    return;
                }
                gfx.egui.run(&gfx.window, |ectx| {
                    ui_action = engine_ui::draw(ectx, model, data, active.as_deref());
                });
                let Gfx {
                    ref ctx,
                    ref mut swapchain,
                    ref renderer,
                    ref mut egui,
                    egui_cmd_pool,
                    egui_queue,
                    ..
                } = *gfx;
                let present = swapchain.present_blit_with_overlay(
                    ctx,
                    renderer.light_output_image(),
                    WINDOW_W,
                    WINDOW_H,
                    ash::vk::Semaphore::null(),
                    |cmd, image_index| {
                        egui.cmd_paint(ctx, cmd, image_index, egui_queue, egui_cmd_pool)
                    },
                );
                if let Err(e) = present {
                    eprintln!("[engine_play] present_blit_with_overlay failed: {e}");
                }
            }

            // Prefer the visible egui's action; fall back to update's headless
            // action (e.g. when running without a window).
            let action = ui_action.or(out.action);
            if let Some(action) = action {
                if self.host.apply_action(&action) {
                    // Restart: re-open from scratch (Wave 1 keeps the same rt;
                    // a real restart re-seeds the runtime — Plan D).
                    self.host.active_screen = None;
                    self.host.last_level = 0;
                    self.host.upgrades.clear();
                }
            }
        }
    }

    /// Map a winit logical key to a stable lowercase name for the held/pressed
    /// sets (matches the sample controls' key names: w/a/s/d, space).
    fn key_to_name(k: &Key) -> Option<String> {
        match k {
            Key::Character(c) => {
                let lower = c.to_lowercase();
                Some(lower)
            }
            Key::Named(NamedKey::Space) => Some("space".into()),
            Key::Named(NamedKey::ArrowUp) => Some("arrowup".into()),
            Key::Named(NamedKey::ArrowDown) => Some("arrowdown".into()),
            Key::Named(NamedKey::ArrowLeft) => Some("arrowleft".into()),
            Key::Named(NamedKey::ArrowRight) => Some("arrowright".into()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::Painted;
    use crate::mock::MockRuntime;

    fn setup() -> (MockRuntime, EngineBridge, HostState, ControlsDescriptor) {
        let rt = MockRuntime::new();
        let render = RenderDescriptor::from_json(rt.render_descriptor()).unwrap();
        let controls = ControlsDescriptor::from_json(rt.controls_descriptor()).unwrap();
        let bridge = EngineBridge::new_headless(render);
        let host = HostState::new(PlayerConfig::default(), mock_ui_model());
        (rt, bridge, host, controls)
    }

    #[test]
    fn player_loop_headless() {
        let (mut rt, bridge, mut host, controls) = setup();
        let ectx = egui::Context::default();
        let mut grid = Painted::new();
        let mut last_cells = Vec::new();

        let mut held = HashSet::new();
        held.insert("d".to_string());
        let pressed = HashSet::new();

        // Drive several synthetic frames; no panic, sim advances.
        let start_tick = rt.tick();
        let mut last_out = UpdateOutput::default();
        for _ in 0..5 {
            last_out = update(
                &mut rt,
                &bridge,
                &mut host,
                &controls,
                &held,
                &pressed,
                &ectx,
                &mut grid,
                &mut last_cells,
            );
        }
        assert_eq!(rt.tick(), start_tick + 5, "sim stepped each frame");

        // Held "d" produced a positive ctl.move_x input on the mock.
        let mx = rt
            .last_input_for("ctl.move_x")
            .expect("ctl.move_x was written");
        assert!(mx > 0.0, "held d → move_x > 0, got {mx}");
        // And the update reported it in its inputs.
        assert!(last_out
            .inputs
            .iter()
            .any(|(f, v)| f == "ctl.move_x" && *v > 0.0));

        // Something got painted (the player + enemy splats).
        assert!(!grid.cells.is_empty(), "bridge painted agent cells");
    }

    #[test]
    fn level_up_opens_menu_and_action_applies() {
        let (mut rt, bridge, mut host, controls) = setup();
        let ectx = egui::Context::default();
        let mut grid = Painted::new();
        let mut last_cells = Vec::new();
        let held = HashSet::new();
        let pressed = HashSet::new();

        // Force the level view above the per-level threshold for the player slot.
        rt.views.insert(("xp".into(), 0), 6.0); // > xp_per_level (5.0) → level 1

        let out = update(
            &mut rt,
            &bridge,
            &mut host,
            &controls,
            &held,
            &pressed,
            &ectx,
            &mut grid,
            &mut last_cells,
        );
        assert_eq!(host.active_screen.as_deref(), Some("level_up"));
        assert!(out.frozen == false, "first frame opens the menu (frozen next frame)");

        // Next frame is frozen; movement zeroed.
        let _ = update(
            &mut rt,
            &bridge,
            &mut host,
            &controls,
            &HashSet::from(["d".to_string()]),
            &pressed,
            &ectx,
            &mut grid,
            &mut last_cells,
        );
        assert_eq!(rt.last_input_for("ctl.move_x"), Some(0.0), "frozen → move zeroed");

        // Applying an Increment raises the counter and closes the menu.
        let restart = host.apply_action(&UiAction::Increment("bolt_level".into()));
        assert!(!restart);
        assert_eq!(host.upgrades.get("bolt_level"), Some(&1.0));
        assert!(host.active_screen.is_none());

        // The closed-menu frame mirrors the upgrade counter into the runtime.
        let _ = update(
            &mut rt,
            &bridge,
            &mut host,
            &controls,
            &held,
            &pressed,
            &ectx,
            &mut grid,
            &mut last_cells,
        );
        assert_eq!(rt.last_input_for("ctl.bolt_level"), Some(1.0));
    }

    #[test]
    fn dead_followed_agent_opens_death_screen() {
        let (rt, bridge, mut host, controls) = setup();
        // Kill the player by removing it from the snapshot: easiest is to set
        // its mana out of the follow band so `followed` returns None.
        // The mock has no mutator, so emulate via a fresh runtime whose player
        // is dead: set alive=false via a custom snapshot is not exposed, so we
        // instead assert the no-followed path opens "dead".
        let ectx = egui::Context::default();
        let mut grid = Painted::new();
        let mut last_cells = Vec::new();
        // Observer-less: drive with an empty held set; the mock player is in
        // band, so to exercise death we craft a runtime with the player dead.
        // Use the public `agents`-less path: a runtime returning no in-band agent.
        struct DeadRt(MockRuntime);
        impl PlayableRuntime for DeadRt {
            fn tick(&self) -> u64 {
                self.0.tick()
            }
            fn step(&mut self) {
                self.0.step()
            }
            fn set_input(&mut self, f: &str, v: f32) {
                self.0.set_input(f, v)
            }
            fn agent_snapshot(&mut self) -> Vec<engine_play_api::AgentView> {
                // Only an enemy (mana 2.0) → no agent in the follow band.
                vec![engine_play_api::AgentView {
                    pos: [3.0, 0.0, 0.0],
                    alive: true,
                    hp: 10.0,
                    mana: 2.0,
                    move_speed: 0.5,
                    creature_type: 1,
                }]
            }
            fn view_value(&mut self, v: &str, s: u32) -> f32 {
                self.0.view_value(v, s)
            }
            fn render_descriptor(&self) -> &'static str {
                self.0.render_descriptor()
            }
            fn controls_descriptor(&self) -> &'static str {
                self.0.controls_descriptor()
            }
            fn ui_descriptor(&self) -> &'static str {
                self.0.ui_descriptor()
            }
        }
        let mut dead = DeadRt(rt);
        let _ = update(
            &mut dead,
            &bridge,
            &mut host,
            &controls,
            &HashSet::new(),
            &HashSet::new(),
            &ectx,
            &mut grid,
            &mut last_cells,
        );
        assert_eq!(host.active_screen.as_deref(), Some("dead"));
    }
}
