# Engine Plan 3.0 — Visualization Harness (visible window from Task 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `cargo run -p viz -- <scenario.toml>` — an interactive windowed viewer that drives `engine::step::step(...)` and renders live agents plus combat events as colored voxels. Task 1 produces a visible Vulkan window (ground plane). By Task 5 the user can spawn, tick, watch attacks, and replay via pause / step / speed / reload controls. Visual feedback is the one tool that catches bugs where test-author and impl-author are the same subagent and collude.

**Architecture:** A new workspace crate `crates/viz/` that templates directly against `/home/ricky/Projects/game/src/world_sim/voxel_app.rs` — no `voxel_engine::app::App` trait, no headless `Scene`, no shadow-buffer machinery. `main()` builds a `winit::event_loop::EventLoop`; an `ApplicationHandler` struct `VizApp` creates the `winit::window::Window` on `resumed()`, constructs `VulkanContext` + `VulkanAllocator` + `SwapchainContext` + `VoxelRenderer` + `FreeCamera`, and renders every `about_to_wait` via one `GpuVoxelTexture` covering a fixed-size (128³) `VoxelGrid`. Each frame: clear the above-ground layer, paint alive agent voxels, paint event overlays (attack lines, death markers, announce rings), re-upload the grid, call `VoxelRenderer::render_frame_gpu(&ctx, &camera, &[(&tex, white, origin, dims)])`, then `SwapchainContext::present_blit(...)`. The sim advances 10 Hz via wall-clock dt accumulation. Keyboard (Space / `.` / R / `[` / `]`) flips flags on `AppState`.

**Tech Stack:** Rust 2021; `voxel_engine` (`path = "/home/ricky/Projects/voxel_engine"`, feature `app-harness` — brings `winit` 0.30); `engine` crate from this workspace; `glam` 0.29; `serde` 1.0 + `toml` 0.8; `anyhow` 1.0.

---

## Files overview

New files under `crates/viz/`:

| Path | Responsibility |
|---|---|
| `Cargo.toml` | Deps on `engine`, `voxel_engine` (feature `app-harness`), `winit`, `glam`, `anyhow`, `serde`, `toml`. |
| `src/main.rs` | Bin entry: parse CLI (one positional scenario path), call `app::run`. |
| `src/lib.rs` | Library facade exposing `grid_paint`, `overlays`, `palette`, `scenario` for tests. |
| `src/app.rs` | `VizApp: ApplicationHandler` — window creation, event dispatch. |
| `src/state.rs` | `AppState` — Vulkan handles, VoxelRenderer, FreeCamera, `SimState` + scratch + events + cascade + backend, overlay tracker, tick accumulator. |
| `src/scenario.rs` | TOML: `Scenario { world: World, agent: Vec<AgentSpec> }`. |
| `src/grid_paint.rs` | Pure VoxelGrid helpers: `paint_ground_plane`, `clear_above_ground`, `paint_agent`, `paint_line`, `paint_ring`, `grid_index_of`. |
| `src/palette.rs` | 256-entry RGBA palette + `PAL_*` index constants; matches colors from `src/world_sim/voxel_bridge.rs` where semantics overlap. |
| `src/overlays.rs` | `Overlay` + `OverlayTracker`: ingest events → paint attack lines / death markers / announce rings with per-kind TTL. |
| `scenarios/viz_basic.toml` | 4 humans + 1 wolf — smoke/approach/attack/die. |
| `scenarios/viz_attack.toml` | 1 human + 1 wolf 3 m apart — attack-line verification. |
| `scenarios/viz_announce.toml` | 1 speaker + 6 listeners — ring overlay verification (requires test backend in a later plan). |
| `tests/scenario_load.rs` | TOML parsing tests. |
| `tests/tick_advances.rs` | Smoke: `engine::step::step` advances `state.tick` and moves the wolf. |
| `tests/grid_paint.rs` | Pure unit tests for `paint_ground_plane`, `clear_above_ground`, `paint_agent`, `paint_line`, `paint_ring`, `grid_index_of`. |

Modified files:

| Path | Change |
|---|---|
| `Cargo.toml` (workspace root) | Add `"crates/viz"` to `[workspace].members`. |
| `docs/engine/status.md` | Update Plan 3.0 row → executed; append "Visual-check checklist" section. |

---

## Task 1: Crate scaffolding + Vulkan init + visible ground plane

**Goal:** `cargo run -p viz` opens a Vulkan window with a gray ground plane and blue sky. WASDQE + right-click-drag moves the camera. Escape exits cleanly. No scenario, no sim, no agents.

**Files:**
- Create: `crates/viz/Cargo.toml`, `src/main.rs`, `src/lib.rs`, `src/app.rs`, `src/state.rs`, `src/palette.rs`, `src/grid_paint.rs`
- Modify: `/home/ricky/Projects/game/.worktrees/world-sim-bench/Cargo.toml`

- [ ] **Step 1: Add `crates/viz` to workspace members**

Modify the workspace `[workspace]` section — keep `exclude` untouched:

```toml
[workspace]
members = [".", "crates/tactical_sim", "crates/engine", "crates/viz"]
exclude = ["crates/world_sim_bench"]
```

- [ ] **Step 2: Write `crates/viz/Cargo.toml`**

```toml
[package]
name = "viz"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[[bin]]
name = "viz"
path = "src/main.rs"

[dependencies]
engine = { path = "../engine" }
voxel_engine = { path = "/home/ricky/Projects/voxel_engine", features = ["app-harness"] }
winit = "0.30"
glam = "0.29"
anyhow = "1"
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
```

- [ ] **Step 3: Write `crates/viz/src/palette.rs`**

```rust
//! Voxel palette. Indices are chosen so humans are blue and wolves red
//! exactly as in `src/world_sim/voxel_bridge.rs`, so visual meaning
//! transfers between the legacy renderer and this viz.

pub const PAL_AIR:      u8 = 0;
pub const PAL_GROUND:   u8 = 1;
pub const PAL_HUMAN:    u8 = 10; // CreatureType::Human  = 0
pub const PAL_WOLF:     u8 = 11; // CreatureType::Wolf   = 1
pub const PAL_DEER:     u8 = 12; // CreatureType::Deer   = 2
pub const PAL_DRAGON:   u8 = 13; // CreatureType::Dragon = 3
pub const PAL_ATTACK:   u8 = 20;
pub const PAL_DEATH:    u8 = 21;
pub const PAL_ANNOUNCE: u8 = 22;

pub fn build_palette_rgba() -> [[u8; 4]; 256] {
    let mut p = [[0u8, 0, 0, 0]; 256];
    p[PAL_GROUND as usize]   = [120, 125, 128, 255]; // matches voxel_bridge::Stone
    p[PAL_HUMAN as usize]    = [ 60, 120, 220, 255]; // matches voxel_bridge::NpcIdle
    p[PAL_WOLF as usize]     = [180,  30,  30, 255]; // matches voxel_bridge::MonsterMarker
    p[PAL_DEER as usize]     = [210, 180, 120, 255];
    p[PAL_DRAGON as usize]   = [220,  80,  20, 255];
    p[PAL_ATTACK as usize]   = [230,  60,  40, 255];
    p[PAL_DEATH as usize]    = [ 10,  10,  10, 255];
    p[PAL_ANNOUNCE as usize] = [240, 245, 250, 255];
    p
}

pub fn creature_palette_index(ct: engine::creature::CreatureType) -> u8 {
    use engine::creature::CreatureType as CT;
    match ct {
        CT::Human  => PAL_HUMAN,
        CT::Wolf   => PAL_WOLF,
        CT::Deer   => PAL_DEER,
        CT::Dragon => PAL_DRAGON,
    }
}
```

- [ ] **Step 4: Write `crates/viz/src/grid_paint.rs`**

```rust
//! Pure voxel-grid helpers. No Vulkan, no sim state.
//! Engine convention: Y-up. 1 voxel = 1 m. Grid is GRID_SIDE³.

use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;

use crate::palette::{PAL_AIR, PAL_GROUND};

pub const GRID_SIDE: u32 = 128;
pub const GROUND_Y:  u32 = 2;

pub fn paint_ground_plane(grid: &mut VoxelGrid) {
    let (w, _h, d) = grid.dimensions();
    for z in 0..d {
        for x in 0..w {
            grid.set(x, GROUND_Y, z, PAL_GROUND);
        }
    }
}

pub fn clear_above_ground(grid: &mut VoxelGrid) {
    let (w, h, d) = grid.dimensions();
    for y in (GROUND_Y + 1)..h {
        for z in 0..d {
            for x in 0..w {
                grid.set(x, y, z, PAL_AIR);
            }
        }
    }
}

pub fn grid_index_of(pos: Vec3, grid: &VoxelGrid) -> Option<(u32, u32, u32)> {
    let (w, h, d) = grid.dimensions();
    let x = pos.x.floor() as i32;
    let y = pos.y.floor() as i32;
    let z = pos.z.floor() as i32;
    if x < 0 || y < 0 || z < 0 { return None; }
    if x as u32 >= w || y as u32 >= h || z as u32 >= d { return None; }
    Some((x as u32, y as u32, z as u32))
}

pub fn paint_agent(grid: &mut VoxelGrid, pos: Vec3, palette_idx: u8) {
    // Lift to sit on top of the ground plane.
    let lifted = Vec3::new(pos.x, pos.y.max((GROUND_Y + 1) as f32), pos.z);
    if let Some((x, y, z)) = grid_index_of(lifted, grid) {
        grid.set(x, y, z, palette_idx);
    }
}

/// 3D Bresenham-ish parametric line. Out-of-bounds voxels skipped.
pub fn paint_line(grid: &mut VoxelGrid, from: Vec3, to: Vec3, palette_idx: u8) {
    let dx = (to.x - from.x).abs();
    let dy = (to.y - from.y).abs();
    let dz = (to.z - from.z).abs();
    let steps = dx.max(dy).max(dz).ceil() as i32;
    if steps <= 0 {
        if let Some((x, y, z)) = grid_index_of(from, grid) {
            grid.set(x, y, z, palette_idx);
        }
        return;
    }
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = from.lerp(to, t);
        if let Some((x, y, z)) = grid_index_of(p, grid) {
            grid.set(x, y, z, palette_idx);
        }
    }
}

/// Axis-aligned ring (constant y) via polar sampling.
pub fn paint_ring(grid: &mut VoxelGrid, center: Vec3, radius: f32, palette_idx: u8) {
    let r = radius.max(0.0);
    let steps = ((2.0 * std::f32::consts::PI * r).ceil() as i32).max(8);
    for i in 0..steps {
        let theta = (i as f32) / (steps as f32) * std::f32::consts::TAU;
        let x = center.x + r * theta.cos();
        let z = center.z + r * theta.sin();
        if let Some((gx, gy, gz)) = grid_index_of(Vec3::new(x, center.y, z), grid) {
            grid.set(gx, gy, gz, palette_idx);
        }
    }
}
```

- [ ] **Step 5: Write `crates/viz/src/state.rs`**

```rust
//! App state. Templated after `src/world_sim/voxel_app.rs::AppState`
//! but far smaller: one GpuVoxelTexture, no chunk pool, no terrain
//! compute, no mega-chunk bookkeeping.

use std::collections::HashSet;
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

use viz::grid_paint::{paint_ground_plane, GRID_SIDE};
use viz::palette::build_palette_rgba;

// Matches src/world_sim/constants.rs for visual consistency with legacy renderer.
pub const RENDER_WIDTH:  u32 =  480;
pub const RENDER_HEIGHT: u32 =  270;
pub const WINDOW_WIDTH:  u32 = 1280;
pub const WINDOW_HEIGHT: u32 =  720;

pub struct AppState {
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

    pub last_frame:  Instant,
}

impl AppState {
    pub fn new(window: Window) -> Result<Self> {
        let ctx = VulkanContext::new_with_surface_extensions(&window)
            .context("create Vulkan context")?;
        let alloc = VulkanAllocator::new(&ctx)
            .context("create Vulkan allocator")?;
        let mut swapchain = SwapchainContext::new(&ctx, &window)
            .context("create swapchain")?;
        let renderer = VoxelRenderer::new(&ctx, RENDER_WIDTH, RENDER_HEIGHT)
            .context("create voxel renderer")?;

        // Pre-record blit cmds for the stable output image — enables the
        // SwapchainContext fast path (matches voxel_app.rs:273).
        swapchain.prepare_blit_commands(
            &ctx, renderer.gbuffer_output_image(),
            RENDER_WIDTH, RENDER_HEIGHT,
        ).context("prepare blit commands")?;

        // Blank dark-blue first frame while startup finishes (matches voxel_app.rs:281).
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.12, 1.0]);

        let mut grid = VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE);
        paint_ground_plane(&mut grid);

        // Camera: 30 m above + 30 m south of the grid center, looking at the ground.
        let cx = (GRID_SIDE as f32) * 0.5;
        let eye = Vec3::new(cx, 40.0, cx - 30.0);
        let target = Vec3::new(cx, 2.0, cx);
        let mut camera = FreeCamera::new(eye, target);
        camera.set_move_speed(20.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            keys_held: HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            grid,
            gpu_texture: None,
            last_frame: Instant::now(),
        })
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
    pub fn render(&mut self) -> Result<()> {
        self.upload_grid()?;
        let tex = self.gpu_texture.as_ref()
            .expect("upload_grid always populates gpu_texture");
        let dims = [GRID_SIDE as f32; 3];
        let position = [0.0f32, 0.0, 0.0];
        let palette_color = [1.0f32, 1.0, 1.0, 1.0];
        let objects: [(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3]); 1] =
            [(tex, palette_color, position, dims)];
        self.renderer
            .render_frame_gpu(&self.ctx, &self.camera, &objects)
            .context("render_frame_gpu")?;
        let src = self.renderer.gbuffer_output_image();
        self.swapchain
            .present_blit(&self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT)
            .context("present_blit")?;
        Ok(())
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed { self.keys_held.insert(key); } else { self.keys_held.remove(&key); }
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
        if let Err(e) = self.render() {
            eprintln!("[viz] render error: {}", e);
        }
    }
}
```

- [ ] **Step 6: Write `crates/viz/src/app.rs`**

```rust
use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::state::{AppState, WINDOW_WIDTH, WINDOW_HEIGHT};

pub struct VizApp {
    pub state: Option<AppState>,
}

impl VizApp {
    pub fn new() -> Self { Self { state: None } }
}

impl ApplicationHandler for VizApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title("World Sim Viz")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => { eprintln!("[viz] create_window failed: {}", e); event_loop.exit(); return; }
        };
        match AppState::new(window) {
            Ok(app) => {
                eprintln!("[viz] Ready. WASDQE moves camera, RMB drag to look, Esc to quit.");
                self.state = Some(app);
            }
            Err(e) => { eprintln!("[viz] AppState::new failed: {}", e); event_loop.exit(); }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let app = match &mut self.state { Some(a) => a, None => return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    app.handle_key(key, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    app.mouse_captured = state == ElementState::Pressed;
                    if !app.mouse_captured { app.last_mouse = None; }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if app.mouse_captured {
                    if let Some((lx, ly)) = app.last_mouse {
                        use voxel_engine::camera::{CameraController, InputState};
                        app.camera.update(&InputState {
                            move_forward: 0.0, move_right: 0.0, move_up: 0.0,
                            mouse_dx: (position.x - lx) as f32,
                            mouse_dy: (position.y - ly) as f32,
                            scroll_delta: 0.0,
                        }, 0.0);
                    }
                    app.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.run_frame();
        }
    }
}

pub fn run() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = VizApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
```

- [ ] **Step 7: Write `crates/viz/src/lib.rs`**

```rust
//! Library facade. Bin (`main.rs`) keeps winit-specific modules (`app`,
//! `state`) local because tests can't drive a real window; the rest
//! lives here for test reach.

pub mod grid_paint;
pub mod palette;
// `overlays` and `scenario` are added in Tasks 2 & 4 — leave out for now.
```

- [ ] **Step 8: Write `crates/viz/src/main.rs`**

```rust
mod app;
mod state;

fn main() -> anyhow::Result<()> {
    app::run()
}
```

- [ ] **Step 9: Build and run**

```bash
cargo run -p viz
```

Expected:
- Window ~1280×720 titled `World Sim Viz` opens.
- A flat gray ground plane fills the lower half; dark blue sky above.
- WASDQE moves the camera; right-click + drag looks around.
- Escape closes; process exits code 0.

Failure triage: solid blue → camera inside the ground or grid upload failed; `VulkanContext::new_with_surface_extensions` panic → `voxel_engine`'s `app-harness` feature isn't on.

- [ ] **Step 10: Commit**

```bash
git add \
  Cargo.toml \
  crates/viz/Cargo.toml \
  crates/viz/src/main.rs \
  crates/viz/src/lib.rs \
  crates/viz/src/app.rs \
  crates/viz/src/state.rs \
  crates/viz/src/palette.rs \
  crates/viz/src/grid_paint.rs
git commit -m "$(cat <<'EOF'
feat(viz): Task 1 — windowed Vulkan harness with ground plane

Opens a 1280x720 window, builds VulkanContext / VulkanAllocator /
SwapchainContext / VoxelRenderer / FreeCamera the same way
src/world_sim/voxel_app.rs does, and renders a single 128^3 VoxelGrid
with a ground plane. WASDQE + right-click drag move the camera.

No sim, no scenario, no agents — wired in Tasks 2–5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Scenario loader + agent spawn (visible agents, no ticking yet)

**Goal:** `cargo run -p viz -- crates/viz/scenarios/viz_basic.toml` opens a window that shows the ground plane *and* one colored voxel per spawned agent. Humans are blue, wolves are red. `SimState` / `SimScratch` / `EventRing` / `CascadeRegistry` / `UtilityBackend` exist; `engine::step::step(...)` is never called — `tick` stays at 0.

**Files:**
- Create: `crates/viz/src/scenario.rs`, `scenarios/viz_basic.toml`, `tests/scenario_load.rs`
- Modify: `src/lib.rs`, `src/main.rs`, `src/app.rs`, `src/state.rs`

- [ ] **Step 1: Write `crates/viz/src/scenario.rs`**

```rust
use std::path::Path;
use anyhow::{bail, Context, Result};
use glam::Vec3;
use serde::Deserialize;

use engine::creature::CreatureType;

#[derive(Debug, Deserialize)]
pub struct Scenario {
    #[serde(default)] pub world: World,
    #[serde(default)] pub agent: Vec<AgentSpec>,
}

#[derive(Debug, Deserialize)]
pub struct World {
    #[serde(default = "default_seed")]      pub seed:      u64,
    #[serde(default = "default_agent_cap")] pub agent_cap: u32,
}
fn default_seed() -> u64 { 42 }
fn default_agent_cap() -> u32 { 64 }

impl Default for World {
    fn default() -> Self { Self { seed: default_seed(), agent_cap: default_agent_cap() } }
}

#[derive(Debug, Deserialize)]
pub struct AgentSpec {
    pub creature_type: String,
    pub pos: [f32; 3],
    #[serde(default = "default_hp")] pub hp: f32,
}
fn default_hp() -> f32 { 100.0 }

impl AgentSpec {
    pub fn creature(&self) -> Result<CreatureType> {
        match self.creature_type.as_str() {
            "Human"  => Ok(CreatureType::Human),
            "Wolf"   => Ok(CreatureType::Wolf),
            "Deer"   => Ok(CreatureType::Deer),
            "Dragon" => Ok(CreatureType::Dragon),
            other => bail!("unknown creature_type {:?} (expected Human/Wolf/Deer/Dragon)", other),
        }
    }
    pub fn position(&self) -> Vec3 { Vec3::new(self.pos[0], self.pos[1], self.pos[2]) }
}

pub fn load<P: AsRef<Path>>(path: P) -> Result<Scenario> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("read {:?}", path))?;
    let s: Scenario = toml::from_str(&text).with_context(|| format!("parse {:?}", path))?;
    Ok(s)
}
```

- [ ] **Step 2: Add `pub mod scenario;` to `crates/viz/src/lib.rs`**

```rust
pub mod grid_paint;
pub mod palette;
pub mod scenario;
// overlays added in Task 4
```

- [ ] **Step 3: Write `crates/viz/scenarios/viz_basic.toml`**

```toml
[world]
seed = 42
agent_cap = 16

[[agent]]
creature_type = "Human"
pos = [60.0, 4.0, 60.0]

[[agent]]
creature_type = "Human"
pos = [68.0, 4.0, 60.0]

[[agent]]
creature_type = "Human"
pos = [60.0, 4.0, 68.0]

[[agent]]
creature_type = "Human"
pos = [68.0, 4.0, 68.0]

[[agent]]
creature_type = "Wolf"
pos = [80.0, 4.0, 80.0]
```

- [ ] **Step 4: Add sim state fields to `AppState` in `src/state.rs`**

Add below `gpu_texture`:

```rust
    pub sim:       engine::state::SimState,
    pub scratch:   engine::step::SimScratch,
    pub events:    engine::event::EventRing,
    pub cascade:   engine::cascade::CascadeRegistry,
    pub backend:   engine::policy::UtilityBackend,
    pub agent_ids: Vec<engine::ids::AgentId>,
```

- [ ] **Step 5: Replace `AppState::new(window)` with a scenario-taking version**

```rust
    pub fn new(window: Window, scenario: viz::scenario::Scenario) -> Result<Self> {
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

        // Camera aims at the spawn centroid; falls back to grid center if empty.
        let center: Vec3 = if scenario.agent.is_empty() {
            Vec3::new((GRID_SIDE as f32) * 0.5, 2.0, (GRID_SIDE as f32) * 0.5)
        } else {
            let sum: Vec3 = scenario.agent.iter().map(|a| a.position()).sum();
            sum / (scenario.agent.len() as f32)
        };
        let eye = Vec3::new(center.x, center.y + 30.0, center.z - 30.0);
        let target = Vec3::new(center.x, 2.0, center.z);
        let mut camera = FreeCamera::new(eye, target);
        camera.set_move_speed(20.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            keys_held: HashSet::new(),
            mouse_captured: false, last_mouse: None,
            grid, gpu_texture: None,
            sim, scratch, events, cascade, backend, agent_ids,
            last_frame: Instant::now(),
        })
    }
```

- [ ] **Step 6: Replace the `render` body so it paints alive agents each frame**

```rust
    pub fn render(&mut self) -> Result<()> {
        viz::grid_paint::clear_above_ground(&mut self.grid);
        for id in self.sim.agents_alive() {
            let pos = match self.sim.agent_pos(id) { Some(p) => p, None => continue };
            let ct  = self.sim.agent_creature_type(id).unwrap_or(engine::creature::CreatureType::Human);
            let idx = viz::palette::creature_palette_index(ct);
            viz::grid_paint::paint_agent(&mut self.grid, pos, idx);
        }
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
```

- [ ] **Step 7: Wire scenario through `app.rs` and `main.rs`**

Replace `VizApp` / `run` in `src/app.rs`:

```rust
pub struct VizApp {
    pub scenario: Option<viz::scenario::Scenario>,
    pub state:    Option<AppState>,
}

impl VizApp {
    pub fn new(scenario: viz::scenario::Scenario) -> Self {
        Self { scenario: Some(scenario), state: None }
    }
}
```

In `resumed`, after `create_window`:

```rust
        let scenario = self.scenario.take().expect("scenario set before resumed");
        match AppState::new(window, scenario) {
            Ok(app) => { eprintln!("[viz] Ready."); self.state = Some(app); }
            Err(e)  => { eprintln!("[viz] AppState::new failed: {}", e); event_loop.exit(); }
        }
```

Replace `pub fn run()`:

```rust
pub fn run(scenario: viz::scenario::Scenario) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = VizApp::new(scenario);
    event_loop.run_app(&mut app)?;
    Ok(())
}
```

Replace `src/main.rs`:

```rust
mod app;
mod state;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <scenario.toml>", args.get(0).map(String::as_str).unwrap_or("viz"));
        std::process::exit(2);
    }
    let scenario = viz::scenario::load(&args[1])?;
    app::run(scenario)
}
```

- [ ] **Step 8: Write `crates/viz/tests/scenario_load.rs`**

```rust
use viz::scenario;

#[test]
fn viz_basic_parses_into_4_humans_and_1_wolf() {
    let s = scenario::load("scenarios/viz_basic.toml").expect("viz_basic parses");
    assert_eq!(s.agent.len(), 5);
    assert_eq!(s.world.seed, 42);
    let humans = s.agent.iter().filter(|a| a.creature_type == "Human").count();
    let wolves = s.agent.iter().filter(|a| a.creature_type == "Wolf").count();
    assert_eq!(humans, 4);
    assert_eq!(wolves, 1);
}

#[test]
fn unknown_creature_type_errors() {
    let spec = scenario::AgentSpec {
        creature_type: "Goblin".into(),
        pos: [0.0, 0.0, 0.0],
        hp: 100.0,
    };
    let err = spec.creature().unwrap_err().to_string();
    assert!(err.contains("Goblin"), "error mentions bad type: {}", err);
}
```

- [ ] **Step 9: Run tests and the binary**

```bash
cargo test -p viz --test scenario_load
```
Expected: 2 passes.

```bash
cargo run -p viz -- crates/viz/scenarios/viz_basic.toml
```
Expected: window opens; ground + 4 blue voxels in a square + 1 red voxel NE; stdout `[viz] Spawned 5 agents (cap 16)`. Agents DO NOT move yet.

- [ ] **Step 10: Commit**

```bash
git add \
  crates/viz/src/lib.rs crates/viz/src/main.rs crates/viz/src/app.rs crates/viz/src/state.rs \
  crates/viz/src/scenario.rs \
  crates/viz/scenarios/viz_basic.toml \
  crates/viz/tests/scenario_load.rs
git commit -m "$(cat <<'EOF'
feat(viz): Task 2 — scenario loader + agent spawn

TOML scenarios: [world] seed/cap + [[agent]] creature_type/pos/hp.
AppState::new now takes a Scenario, constructs SimState + scratch +
ring + cascade + UtilityBackend, and spawns each agent via
sim.spawn_agent. Render path clears the above-ground layer every
frame and paints alive agents as single voxels colored by
CreatureType (humans blue, wolves red — matches voxel_bridge.rs).

No ticking yet; agents stay put.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Tick loop — agents move

**Goal:** Each frame accumulates wall-clock `dt`; when it crosses `TICK_PERIOD = 0.1 s` (10 Hz), call `engine::step::step(...)` once. The `UtilityBackend` will emit `MoveToward` / `Attack`, so the wolf walks toward the human square and (inside 2 m range) deals 10 HP/tick damage. Visible: agents move; dead humans' voxels vanish (alive-only render loop).

**Files:**
- Modify: `crates/viz/src/state.rs`
- Create: `crates/viz/tests/tick_advances.rs`

- [ ] **Step 1: Add tick-accumulator fields to `AppState`**

```rust
    pub tick_period:         f32,
    pub sim_accum:           f32,
    pub sim_speed:           f32,
    pub paused:              bool,
    pub max_ticks_per_frame: u32,
```

Init values in `AppState::new`:

```rust
    tick_period: 0.1,
    sim_accum:   0.0,
    sim_speed:   1.0,
    paused:      false,
    max_ticks_per_frame: 8,
```

- [ ] **Step 2: Implement `tick_sim` on `AppState`**

```rust
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
```

- [ ] **Step 3: Call it from `run_frame`**

```rust
    pub fn run_frame(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_camera(dt);
        let _n_ticks = self.tick_sim(dt);
        if let Err(e) = self.render() {
            eprintln!("[viz] render error: {}", e);
        }
    }
```

- [ ] **Step 4: Write `crates/viz/tests/tick_advances.rs`**

```rust
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn step_advances_tick() {
    let mut sim = SimState::new(4, 42);
    let mut scratch = SimScratch::new(4);
    let mut events  = EventRing::with_cap(64);
    let cascade = CascadeRegistry::new();
    let backend = UtilityBackend;

    sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(3.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    assert_eq!(sim.tick, 0);
    for expected in 1..=20u32 {
        step(&mut sim, &mut scratch, &mut events, &backend, &cascade);
        assert_eq!(sim.tick, expected, "tick should advance by 1 per step");
    }
}

#[test]
fn wolf_moves_toward_human_across_20_ticks() {
    let mut sim = SimState::new(4, 42);
    let mut scratch = SimScratch::new(4);
    let mut events  = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();
    let backend = UtilityBackend;

    let human = sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let wolf = sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(20.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    let wolf_start = sim.agent_pos(wolf).unwrap();
    for _ in 0..20 {
        step(&mut sim, &mut scratch, &mut events, &backend, &cascade);
    }
    let wolf_end = sim.agent_pos(wolf).unwrap();
    let human_pos = sim.agent_pos(human).unwrap_or(Vec3::ZERO);
    let d_before = wolf_start.distance(human_pos);
    let d_after  = wolf_end.distance(human_pos);
    assert!(
        d_after < d_before,
        "wolf should have closed distance — before={:.2} after={:.2}", d_before, d_after,
    );
}
```

- [ ] **Step 5: Run tests and binary**

```bash
cargo test -p viz --test tick_advances
```
Expected: 2 passes.

```bash
cargo run -p viz -- crates/viz/scenarios/viz_basic.toml
```
Expected: wolf voxel visibly walks toward the human cluster over ~2 s; after ~10 s one or more blue voxels disappear (dead humans).

- [ ] **Step 6: Commit**

```bash
git add crates/viz/src/state.rs crates/viz/tests/tick_advances.rs
git commit -m "$(cat <<'EOF'
feat(viz): Task 3 — drive engine::step from wall-clock dt at 10 Hz

AppState::tick_sim buckets accumulated dt and calls engine::step::step
once per 100 ms of unpaused time (scaled by sim_speed). Caps bursts at
8 ticks/frame to avoid death-spirals. Wolf now visibly walks toward
humans and kills them over ~10 s.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Event overlays — attack lines, death markers, announce rings

**Goal:** `AgentAttacked` events paint a red voxel line attacker→target (5-tick TTL). `AgentDied` events drop a permanent black voxel. `AnnounceEmitted` paints an expanding white ring around the speaker (3-tick TTL, radius ramps 0→80 m). Overlays paint *after* agents so they win draw priority.

**Files:**
- Create: `crates/viz/src/overlays.rs`, `scenarios/viz_attack.toml`, `scenarios/viz_announce.toml`, `tests/grid_paint.rs`
- Modify: `crates/viz/src/lib.rs`, `crates/viz/src/state.rs`

- [ ] **Step 1: Write `crates/viz/src/overlays.rs`**

```rust
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
```

- [ ] **Step 2: Export `overlays` from `src/lib.rs`**

```rust
pub mod grid_paint;
pub mod overlays;
pub mod palette;
pub mod scenario;
```

- [ ] **Step 3: Add an `OverlayTracker` to `AppState`**

Add field:

```rust
    pub overlays: viz::overlays::OverlayTracker,
```

Init in `AppState::new`:

```rust
    overlays: viz::overlays::OverlayTracker::new(),
```

Replace `render` body with the overlay-aware version:

```rust
    pub fn render(&mut self) -> Result<()> {
        viz::grid_paint::clear_above_ground(&mut self.grid);

        self.overlays.ingest_with_state(&self.events, &self.sim);
        self.overlays.prune(self.sim.tick);

        for id in self.sim.agents_alive() {
            let pos = match self.sim.agent_pos(id) { Some(p) => p, None => continue };
            let ct  = self.sim.agent_creature_type(id).unwrap_or(engine::creature::CreatureType::Human);
            let idx = viz::palette::creature_palette_index(ct);
            viz::grid_paint::paint_agent(&mut self.grid, pos, idx);
        }
        self.overlays.paint_into(&mut self.grid, self.sim.tick);

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
```

- [ ] **Step 4: Write `crates/viz/scenarios/viz_attack.toml`**

```toml
[world]
seed = 1
agent_cap = 4

[[agent]]
creature_type = "Human"
pos = [64.0, 4.0, 64.0]

[[agent]]
creature_type = "Wolf"
pos = [67.0, 4.0, 64.0]
```

- [ ] **Step 5: Write `crates/viz/scenarios/viz_announce.toml`**

Note: the default `UtilityBackend` does not emit `MacroAction::Announce`, so the ring overlay only becomes visible once a test backend is wired in (future plan). Listeners are placed so some sit inside the 30 m overhear radius and some outside the 80 m audience radius — ready for that plan.

```toml
[world]
seed = 7
agent_cap = 8

[[agent]]
creature_type = "Human"
pos = [64.0, 4.0, 64.0]

[[agent]]
creature_type = "Human"
pos = [78.0, 4.0, 64.0]   # 14 m east — inside overhear (30 m)

[[agent]]
creature_type = "Human"
pos = [50.0, 4.0, 64.0]   # 14 m west — inside overhear

[[agent]]
creature_type = "Human"
pos = [64.0, 4.0, 94.0]   # 30 m north — at overhear boundary

[[agent]]
creature_type = "Human"
pos = [64.0, 4.0, 34.0]   # 30 m south — at overhear boundary

[[agent]]
creature_type = "Human"
pos = [104.0, 4.0, 64.0]  # 40 m east — outside overhear, inside audience

[[agent]]
creature_type = "Human"
pos = [24.0, 4.0, 64.0]   # 40 m west — outside overhear, inside audience
```

- [ ] **Step 6: Write `crates/viz/tests/grid_paint.rs`**

```rust
use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;
use viz::grid_paint::{
    clear_above_ground, grid_index_of, paint_agent, paint_ground_plane,
    paint_line, paint_ring, GRID_SIDE, GROUND_Y,
};
use viz::palette::{PAL_AIR, PAL_ATTACK, PAL_GROUND, PAL_HUMAN};

fn fresh_grid() -> VoxelGrid { VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE) }

#[test]
fn ground_plane_fills_its_layer() {
    let mut g = fresh_grid();
    paint_ground_plane(&mut g);
    for x in [0u32, 1, GRID_SIDE - 1] {
        for z in [0u32, 1, GRID_SIDE - 1] {
            assert_eq!(g.get(x, GROUND_Y, z), Some(PAL_GROUND));
        }
    }
    assert_eq!(g.get(5, GROUND_Y + 1, 5), Some(PAL_AIR));
    assert_eq!(g.get(5, GROUND_Y - 1, 5), Some(PAL_AIR));
}

#[test]
fn clear_above_ground_erases_only_above_layer() {
    let mut g = fresh_grid();
    paint_ground_plane(&mut g);
    g.set(5, GROUND_Y + 1, 5, PAL_HUMAN);
    g.set(5, GROUND_Y - 1, 5, PAL_GROUND);
    clear_above_ground(&mut g);
    assert_eq!(g.get(5, GROUND_Y + 1, 5), Some(PAL_AIR));
    assert_eq!(g.get(5, GROUND_Y, 5),     Some(PAL_GROUND));
    assert_eq!(g.get(5, GROUND_Y - 1, 5), Some(PAL_GROUND));
}

#[test]
fn paint_agent_stamps_one_voxel() {
    let mut g = fresh_grid();
    paint_agent(&mut g, Vec3::new(10.5, 4.0, 20.0), PAL_HUMAN);
    assert_eq!(g.get(10, 4, 20), Some(PAL_HUMAN));
    assert_eq!(g.get(11, 4, 20), Some(PAL_AIR));
}

#[test]
fn paint_line_draws_3_4_5_triangle() {
    let mut g = fresh_grid();
    paint_line(&mut g,
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(3.0, 10.0, 4.0),
        PAL_ATTACK);
    assert_eq!(g.get(0, 10, 0), Some(PAL_ATTACK));
    assert_eq!(g.get(3, 10, 4), Some(PAL_ATTACK));
    let mut count = 0;
    for x in 0..GRID_SIDE { for z in 0..GRID_SIDE {
        if g.get(x, 10, z) == Some(PAL_ATTACK) { count += 1; }
    }}
    assert!(count >= 4, "expected >=4 line voxels, got {}", count);
}

#[test]
fn paint_ring_stays_on_the_circle() {
    let mut g = fresh_grid();
    let center = Vec3::new(40.0, 10.0, 40.0);
    paint_ring(&mut g, center, 10.0, PAL_ATTACK);
    let mut any = false;
    for x in 0..GRID_SIDE { for z in 0..GRID_SIDE {
        if g.get(x, 10, z) == Some(PAL_ATTACK) {
            let dx = x as f32 - 40.0;
            let dz = z as f32 - 40.0;
            let d  = (dx*dx + dz*dz).sqrt();
            assert!((d - 10.0).abs() < 1.5, "ring voxel at ({},{}) d={:.2}", x, z, d);
            any = true;
        }
    }}
    assert!(any, "paint_ring stamped zero voxels");
}

#[test]
fn grid_index_of_rejects_out_of_bounds() {
    let g = fresh_grid();
    assert!(grid_index_of(Vec3::new(-0.1, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(GRID_SIDE as f32 + 0.1, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(0.0, -1.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(GRID_SIDE as f32, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new((GRID_SIDE - 1) as f32, 0.0, 0.0), &g).is_some());
}
```

- [ ] **Step 7: Run tests and the attack scenario**

```bash
cargo test -p viz
```
Expected: all tests pass (2 scenario_load + 2 tick_advances + 6 grid_paint).

```bash
cargo run -p viz -- crates/viz/scenarios/viz_attack.toml
```
Expected: red wolf closes 3 m in ~3 ticks; red attack line pulses each tick for ~1 s; blue human voxel disappears; single black voxel remains at the death position; wolf idles.

- [ ] **Step 8: Commit**

```bash
git add \
  crates/viz/src/overlays.rs crates/viz/src/lib.rs crates/viz/src/state.rs \
  crates/viz/scenarios/viz_attack.toml crates/viz/scenarios/viz_announce.toml \
  crates/viz/tests/grid_paint.rs
git commit -m "$(cat <<'EOF'
feat(viz): Task 4 — event overlays (attack, death, announce)

OverlayTracker scans EventRing::iter every frame, converts
AgentAttacked / AgentDied / AnnounceEmitted events into Overlay records
with per-kind TTL (attack 5 ticks, death permanent, announce 3 ticks),
prunes expired, and paints them into the VoxelGrid after the agent
paint pass so overlays win draw priority.

Attack → red line attacker→target. Death → black voxel, permanent.
Announce → white expanding ring (only visible once a test backend
emits AnnounceEmitted; UtilityBackend does not).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Controls + HUD + status.md visual checklist

**Goal:** Keyboard controls for pause / single-step / reload / speed. 1 Hz stdout HUD. `docs/engine/status.md` updated with a concrete "Visual-check checklist" section.

**Files:**
- Modify: `crates/viz/src/state.rs`, `src/app.rs`, `src/main.rs`, `docs/engine/status.md`

- [ ] **Step 1: Add HUD + reload fields to `AppState`**

```rust
    pub scenario_path:    std::path::PathBuf,
    pub hud_timer:        Instant,
    pub frames_since_hud: u32,
```

Pass `scenario_path: std::path::PathBuf` as a new arg to `AppState::new`. Init the new fields:

```rust
    scenario_path,
    hud_timer: Instant::now(),
    frames_since_hud: 0,
```

- [ ] **Step 2: Implement `reload_scenario`**

```rust
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
```

- [ ] **Step 3: Expand `handle_key`**

```rust
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
```

- [ ] **Step 4: Implement `maybe_emit_hud` and call it from `run_frame`**

```rust
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

    pub fn run_frame(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_camera(dt);
        let _n = self.tick_sim(dt);
        if let Err(e) = self.render() { eprintln!("[viz] render error: {}", e); }
        self.maybe_emit_hud();
    }
```

- [ ] **Step 5: Plumb `scenario_path` through `app.rs` + `main.rs`**

Update `VizApp`:

```rust
pub struct VizApp {
    pub scenario:      Option<viz::scenario::Scenario>,
    pub scenario_path: Option<std::path::PathBuf>,
    pub state:         Option<AppState>,
}

impl VizApp {
    pub fn new(scenario: viz::scenario::Scenario, scenario_path: std::path::PathBuf) -> Self {
        Self { scenario: Some(scenario), scenario_path: Some(scenario_path), state: None }
    }
}
```

Update `resumed` body (after `create_window`):

```rust
        let scenario = self.scenario.take().expect("scenario set before resumed");
        let scenario_path = self.scenario_path.take().expect("scenario_path set before resumed");
        match AppState::new(window, scenario, scenario_path) {
            Ok(app) => {
                eprintln!("[viz] Ready. Controls:");
                eprintln!("       WASDQE      — move camera");
                eprintln!("       RMB drag    — look");
                eprintln!("       Space       — pause/resume");
                eprintln!("       .           — single step (paused only)");
                eprintln!("       R           — reload scenario from disk");
                eprintln!("       [ / ]       — halve/double sim speed");
                eprintln!("       Esc         — quit");
                self.state = Some(app);
            }
            Err(e) => { eprintln!("[viz] AppState::new failed: {}", e); event_loop.exit(); }
        }
```

Update `pub fn run`:

```rust
pub fn run(scenario: viz::scenario::Scenario, scenario_path: std::path::PathBuf) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = VizApp::new(scenario, scenario_path);
    event_loop.run_app(&mut app)?;
    Ok(())
}
```

Update `src/main.rs`:

```rust
mod app;
mod state;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <scenario.toml>", args.get(0).map(String::as_str).unwrap_or("viz"));
        std::process::exit(2);
    }
    let path = std::path::PathBuf::from(&args[1]);
    let scenario = viz::scenario::load(&path)?;
    app::run(scenario, path)
}
```

- [ ] **Step 6: Update `docs/engine/status.md`**

In the "Plans index" table, replace the Plan 3.0 row:

```markdown
| Plan 3.0 viz harness | `docs/superpowers/plans/2026-04-19-engine-plan-3_0-viz-harness.md` | ✅ executed (Tasks 1–5) |
```

Append this section at the end of the file (after "References"):

```markdown
## Visual-check checklist

Each item is a live acceptance criterion for the Plan 3.0 viz harness.
Run `cargo run -p viz -- <scenario>` and eyeball the result.

| # | Scenario | Expected visual | Catches regression in |
|---|---|---|---|
| V1 | `crates/viz/scenarios/viz_basic.toml` | Ground + 4 blue voxels in an 8 m square + 1 red voxel 20 m NE; the red voxel walks toward the blue cluster over ~2 s. | Move action (§9), nearest-enemy utility scoring. |
| V2 | `viz_basic.toml`, after ~10 s | One or more blue voxels have disappeared; a black voxel persists where each died. | Attack damage + AgentDied emission. |
| V3 | `viz_attack.toml` | Red voxel closes 3 m in ~3 ticks, then a short red line pulses between attacker and target every tick until the human dies. | ATTACK_RANGE = 2.0 m, attack-line overlay ingest. |
| V4 | `viz_attack.toml` post-death | Single black voxel at the former human position; wolf idle (no more targets). | AgentDied cleanup, mask pruning of dead targets. |
| V5 | `viz_announce.toml` + test backend that emits `MacroAction::Announce` (future plan) | White ring expands from speaker over 3 ticks, covering 80 m. Listeners inside 30 m get memories (check events log). | Announce audience enumeration + overhear scan (§10). |
| V6 | Any scenario, paused, pressing `.` | Tick advances by 1 per press; HUD prints `tick={n+1}`. | Pause/step determinism, no accumulator leak. |
| V7 | Any scenario, pressing `]` 4 times | HUD prints `speed=16.00x`; agents move visibly faster; fps ≥ 30. | Tick accumulator math, burst-cap behavior. |
| V8 | Any scenario, pressing `R` | Agents snap back to spawn positions; HUD reports `tick=0`; overlays clear. | Reload path cleans sim + overlays. |

Known gaps:
- V5 requires a test backend that emits `Announce`; `UtilityBackend`
  only emits the 7 implemented micros. Becomes live when a future plan
  wires an announce-enabled policy.
- No in-window HUD — HUD is stdout-only. A later plan can layer egui
  or a text-shader overlay; deliberately out of scope for Plan 3.0.
```

- [ ] **Step 7: Manual verification run**

```bash
cargo run -p viz -- crates/viz/scenarios/viz_basic.toml
```

Run through V1–V8 from the checklist above. All must behave as described.

- [ ] **Step 8: Commit**

```bash
git add \
  crates/viz/src/state.rs crates/viz/src/app.rs crates/viz/src/main.rs \
  docs/engine/status.md
git commit -m "$(cat <<'EOF'
feat(viz): Task 5 — controls, 1 Hz HUD, visual checklist in status.md

Keyboard: Space toggles pause, `.` single-steps while paused, R
reloads the scenario from disk (preserves Vulkan + camera), `[`/`]`
halve/double sim_speed (1/16x .. 32x).

Stdout HUD every 1 s: RUNNING/PAUSED, tick, alive/total, speed, fps,
overlay count.

docs/engine/status.md:
  - Plan 3.0 row → executed
  - New "Visual-check checklist" section with V1–V8 manual acceptance
    criteria bound to specific scenarios.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

1. **Spec coverage:** The user's 5-task breakdown is preserved 1:1 — scaffold/window → spawn → tick → overlays → controls/HUD/status. Nothing dropped.
2. **No placeholders:** Every code block compiles as written. Repeated code (e.g. `AppState::new` body, `resumed` body) is spelled out rather than "same as Task N".
3. **Type consistency:** `SimState`, `SimScratch`, `EventRing::with_cap`, `CascadeRegistry::new`, `UtilityBackend`, `AgentSpawn`, `CreatureType::{Human,Wolf,Deer,Dragon}`, `Event::{AgentMoved,AgentAttacked,AgentDied,AnnounceEmitted}`, `agents_alive`, `agent_pos`, `agent_creature_type` — all match the actual paths in `crates/engine/src/{state,step,event,cascade,policy,creature,ids}`. `VoxelRenderer::new`, `VoxelRenderer::render_frame_gpu`, `VoxelRenderer::gbuffer_output_image`, `SwapchainContext::{new,prepare_blit_commands,present_blit,present_cleared_frame}`, `voxel_gpu::upload_grid_to_gpu`, `VoxelGrid::{new,set,get,dimensions}`, `FreeCamera::{new,set_move_speed,eye_position}`, `CameraController::update(&InputState,dt)` — all match `/home/ricky/Projects/voxel_engine/src/`. The `app-harness` feature gates `VulkanContext::new_with_surface_extensions`, `SwapchainContext::new`, etc. — enabled via `features = ["app-harness"]` in the viz `Cargo.toml`.
4. **Workspace ordering:** workspace `members` edit comes first. Every later `cargo run -p viz` has the crate wired.
5. **Tests:** Each task ships tests proportional to the work (Task 1: visual smoke, Task 2: scenario parse + bad-creature-type error, Task 3: tick advance + wolf-closes-distance, Task 4: 6 pure grid-paint units).

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-engine-plan-3_0-viz-harness.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — run tasks in this session via `superpowers:executing-plans` with checkpoints.

**Which approach?**
