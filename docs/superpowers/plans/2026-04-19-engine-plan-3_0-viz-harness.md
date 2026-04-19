# Engine Plan 3.0 — Visualization Harness (via voxel_engine)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an interactive viewer (`cargo run -p viz -- <scenario.toml>`) that opens a window, drives the `engine` tick loop, and shows agents + combat events live as voxel markers — so the human can eyeball obviously-wrong sim behavior in seconds. Subagent-authored unit tests are self-consistent (same subagent writes test + impl → passes even with real bugs); a running visual lets wrongness surface *independent* of the test author. This plan MUST land before GPU kernel porting (Plan 6+), because GPU bugs are especially hard to catch without a visual feedback loop.

**Architecture:** New workspace member `crates/viz` (cleaner than a root binary — the root crate already owns `xtask` and has heavy deps; `viz` should be a thin presentation layer). The crate defines a `VizApp` struct that implements `voxel_engine::app::App`. `setup()` reads a TOML scenario (`Vec<AgentSpawn>`), constructs `SimState` + `SimScratch` + `EventRing` + `CascadeRegistry`, and spawns agents. `tick(scene, dt)` accumulates wall-clock dt and calls `engine::step::step(...)` once per 100 ms of un-paused time; between ticks, agents are re-rendered as single-voxel markers in a dedicated "viz" scene entity (clear → redraw each frame). Keyboard input (Space/./R/[/]) flips flags on `VizApp`. **Known voxel_engine gap: there is no `run_app(cfg, app)` function that wires winit + the Vulkan renderer. For this MVP we wire winit ourselves in `main()` against a `Scene::new_headless` and call `App::setup`/`App::tick`/`App::on_input` by hand — the window stays blank (no actual rendering yet), and HUD is printed to stdout. A follow-up plan (3.1) will either add a proper `voxel_engine::app::run_app` wrapper around `VoxelRenderer`, or port the viz to render through the existing `VulkanContext`/`VoxelRenderer`.** Agents are rendered as single-voxel markers colored by `CreatureType`; swap for proper entity sprites in Plan 3.1.

**Tech Stack:** Rust 2021, `voxel_engine` (feature `app-harness` — brings in `winit` 0.30 + `egui`), `engine` crate (this workspace), `glam` 0.29, `serde` + `toml` for scenario loading, `anyhow`. No new top-level deps beyond what's already in the workspace lockfile.

---

## Files overview

New:
- `crates/viz/Cargo.toml` — viz crate manifest
- `crates/viz/src/main.rs` — `main()` + winit event loop driver
- `crates/viz/src/app.rs` — `VizApp` struct + `App` trait impl
- `crates/viz/src/scenario.rs` — TOML scenario loader
- `crates/viz/src/render.rs` — agent → voxel marker rendering; event overlay ring buffer
- `crates/viz/src/hud.rs` — stdout HUD (1 Hz tick counter + alive count)
- `crates/viz/scenarios/viz_basic.toml` — 4 humans + 1 wolf scenario
- `crates/viz/tests/scenario_load.rs` — parser test
- `crates/viz/tests/tick_advances.rs` — smoke: viz app drives state.tick forward

Modified:
- `Cargo.toml` (workspace root) — add `crates/viz` to `[workspace].members`
- `docs/engine/status.md` — add/populate "Visual-check checklist" section (Task 8)

---

## Task 1: Viz crate scaffolding + workspace wiring + empty winit loop

**Files:**
- Create: `crates/viz/Cargo.toml`
- Create: `crates/viz/src/main.rs`
- Modify: `/home/ricky/Projects/game/.worktrees/world-sim-bench/Cargo.toml` (workspace members line)

- [ ] **Step 1: Add viz to workspace members**

Modify `/home/ricky/Projects/game/.worktrees/world-sim-bench/Cargo.toml` line 2:

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

[[bin]]
name = "viz"
path = "src/main.rs"

[dependencies]
engine = { path = "../engine" }
voxel_engine = { path = "/home/ricky/Projects/voxel_engine", features = ["app-harness"] }
winit = "0.30"
glam = "0.29"
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
anyhow = "1"
```

- [ ] **Step 3: Write stub `crates/viz/src/main.rs` that opens a window and exits on close**

```rust
//! Visualization harness entry point. Opens a winit window, constructs a
//! headless voxel_engine `Scene`, drives a `voxel_engine::app::App`
//! implementation via a hand-rolled event loop (voxel_engine does not yet
//! ship `run_app(cfg, app)` — see Plan 3.1 follow-up).
//!
//! This task (Task 1) is the scaffold: window opens, closes cleanly on
//! `CloseRequested`, no sim, no rendering.

use anyhow::Result;
use std::time::Instant;
use voxel_engine::app::{App, AppConfig};
use voxel_engine::scene::Scene;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct StubApp;

impl App for StubApp {
    fn setup(&mut self, _scene: &mut Scene) -> Result<()> { Ok(()) }
    fn tick(&mut self, _scene: &mut Scene, _dt: f32) {}
    fn on_input(&mut self, _scene: &mut Scene, _event: &WindowEvent) {}
}

struct Driver<A: App> {
    cfg:          AppConfig,
    app:          A,
    scene:        Scene,
    window:       Option<Window>,
    last_tick:    Instant,
    setup_called: bool,
}

impl<A: App> Driver<A> {
    fn new(cfg: AppConfig, app: A) -> Self {
        let scene = Scene::new_headless(voxel_engine::scene::SceneConfig::default());
        Self {
            cfg,
            app,
            scene,
            window: None,
            last_tick: Instant::now(),
            setup_called: false,
        }
    }
}

impl<A: App> ApplicationHandler for Driver<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title(self.cfg.window_title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.cfg.width as f64,
                self.cfg.height as f64,
            ));
        let w = event_loop.create_window(attrs).expect("create window");
        self.window = Some(w);
        if !self.setup_called {
            self.app.setup(&mut self.scene).expect("App::setup failed");
            self.setup_called = true;
            self.last_tick = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        self.app.on_input(&mut self.scene, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_tick).as_secs_f32();
                self.last_tick = now;
                self.app.tick(&mut self.scene, dt);
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let cfg = AppConfig {
        window_title: "World Sim Viz".to_string(),
        width: 1280,
        height: 720,
        ..Default::default()
    };
    let app = StubApp;
    let event_loop = EventLoop::new()?;
    let mut driver = Driver::new(cfg, app);
    event_loop.run_app(&mut driver)?;
    Ok(())
}
```

- [ ] **Step 4: Build and run stub**

Run: `cargo build -p viz`
Expected: success.

Run: `cargo run -p viz` (manually close the window with the window-manager close button).
Expected: a blank window opens, closes cleanly with exit code 0.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/viz/Cargo.toml crates/viz/src/main.rs
git commit -m "feat(viz): scaffold viz crate + winit event loop driver (Task 1/8)"
```

---

## Task 2: TOML scenario loader

**Files:**
- Create: `crates/viz/src/scenario.rs`
- Create: `crates/viz/tests/scenario_load.rs`
- Modify: `crates/viz/src/main.rs` (declare `mod scenario;`)

- [ ] **Step 1: Write the failing test**

```rust
// crates/viz/tests/scenario_load.rs
use viz::scenario::{load_scenario_str, CreatureKind};

#[test]
fn parses_humans_and_wolves() {
    let toml = r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0

        [[agent]]
        creature_type = "Wolf"
        pos = [5.0, 0.0, 5.0]
        hp = 80.0
    "#;
    let scen = load_scenario_str(toml).unwrap();
    assert_eq!(scen.agents.len(), 2);
    assert_eq!(scen.agents[0].creature_type, CreatureKind::Human);
    assert_eq!(scen.agents[0].pos, [0.0, 0.0, 0.0]);
    assert_eq!(scen.agents[0].hp, 100.0);
    assert_eq!(scen.agents[1].creature_type, CreatureKind::Wolf);
}

#[test]
fn hp_defaults_to_100() {
    let toml = r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
    "#;
    let scen = load_scenario_str(toml).unwrap();
    assert_eq!(scen.agents[0].hp, 100.0);
}

#[test]
fn rejects_unknown_creature_type() {
    let toml = r#"
        [[agent]]
        creature_type = "Goblin"
        pos = [0.0, 0.0, 0.0]
    "#;
    assert!(load_scenario_str(toml).is_err());
}
```

- [ ] **Step 2: Run test, expect compile failure**

Run: `cargo test -p viz --test scenario_load`
Expected: FAIL — `viz::scenario` module does not exist yet.

- [ ] **Step 3: Write `crates/viz/src/scenario.rs`**

```rust
//! TOML scenario loader. Converts a `[[agent]]`-style TOML document into a
//! `Vec<AgentSpawnSpec>` the VizApp feeds into `SimState::spawn_agent`.

use anyhow::{Context, Result};
use engine::creature::CreatureType;
use engine::state::AgentSpawn;
use glam::Vec3;
use serde::Deserialize;
use std::path::Path;

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
pub enum CreatureKind {
    Human,
    Wolf,
    Deer,
    Dragon,
}

impl CreatureKind {
    pub fn to_engine(self) -> CreatureType {
        match self {
            CreatureKind::Human  => CreatureType::Human,
            CreatureKind::Wolf   => CreatureType::Wolf,
            CreatureKind::Deer   => CreatureType::Deer,
            CreatureKind::Dragon => CreatureType::Dragon,
        }
    }
}

fn default_hp() -> f32 { 100.0 }

#[derive(Clone, Debug, Deserialize)]
pub struct AgentSpawnSpec {
    pub creature_type: CreatureKind,
    pub pos:           [f32; 3],
    #[serde(default = "default_hp")]
    pub hp:            f32,
}

impl AgentSpawnSpec {
    pub fn to_engine(&self) -> AgentSpawn {
        AgentSpawn {
            creature_type: self.creature_type.to_engine(),
            pos:           Vec3::new(self.pos[0], self.pos[1], self.pos[2]),
            hp:            self.hp,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Scenario {
    #[serde(rename = "agent", default)]
    pub agents: Vec<AgentSpawnSpec>,
}

pub fn load_scenario_str(src: &str) -> Result<Scenario> {
    toml::from_str::<Scenario>(src)
        .with_context(|| "parsing scenario TOML")
}

pub fn load_scenario_path<P: AsRef<Path>>(path: P) -> Result<Scenario> {
    let path = path.as_ref();
    let src = std::fs::read_to_string(path)
        .with_context(|| format!("reading scenario file {}", path.display()))?;
    load_scenario_str(&src)
}
```

- [ ] **Step 4: Expose module + create lib target**

We want `cargo test -p viz` to find `viz::scenario`. Simplest: add a `lib.rs` alongside `main.rs`. Modify `crates/viz/Cargo.toml` — add a `[lib]` target:

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
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
anyhow = "1"
```

Create `crates/viz/src/lib.rs`:

```rust
//! Visualization harness library. `main.rs` depends on this crate's public
//! modules; tests import from the crate root (`viz::scenario::...`).

pub mod scenario;
```

- [ ] **Step 5: Run test, expect pass**

Run: `cargo test -p viz --test scenario_load`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/viz/Cargo.toml crates/viz/src/lib.rs crates/viz/src/scenario.rs crates/viz/tests/scenario_load.rs
git commit -m "feat(viz): TOML scenario loader with creature-type enum (Task 2/8)"
```

---

## Task 3: `VizApp` — load scenario, drive `engine::step::step` at fixed 100 ms

**Files:**
- Create: `crates/viz/src/app.rs`
- Create: `crates/viz/tests/tick_advances.rs`
- Modify: `crates/viz/src/lib.rs` (add `pub mod app;`)
- Modify: `crates/viz/src/main.rs` (swap `StubApp` for `VizApp`, accept scenario path arg)

- [ ] **Step 1: Write the failing smoke test**

```rust
// crates/viz/tests/tick_advances.rs
//
// Smoke: VizApp, given a minimal scenario, advances state.tick by roughly
// (wall-clock-elapsed / 100 ms) ticks. Runs without a window — the App trait
// is independent of winit.

use viz::app::VizApp;
use viz::scenario::Scenario;
use voxel_engine::app::App;
use voxel_engine::scene::{Scene, SceneConfig};

fn tiny_scenario() -> Scenario {
    toml::from_str(r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0

        [[agent]]
        creature_type = "Human"
        pos = [3.0, 0.0, 3.0]
        hp = 100.0
    "#).unwrap()
}

#[test]
fn setup_spawns_all_agents() {
    let mut app = VizApp::from_scenario(tiny_scenario(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    assert_eq!(app.alive_count(), 2);
    assert_eq!(app.current_tick(), 0);
}

#[test]
fn two_seconds_of_ticks_advances_20_sim_ticks() {
    let mut app = VizApp::from_scenario(tiny_scenario(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    // Feed 2.0 s of wall-clock dt in 50 ms chunks (40 chunks × 0.05 s = 2 s).
    for _ in 0..40 {
        app.tick(&mut scene, 0.05);
    }
    // 100 ms tick rate → 20 sim ticks over 2 s, ±1 for accumulator edge.
    assert!(
        (19..=21).contains(&app.current_tick()),
        "expected ~20 sim ticks, got {}",
        app.current_tick()
    );
}

#[test]
fn pause_stops_tick_advance() {
    let mut app = VizApp::from_scenario(tiny_scenario(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    app.set_paused(true);
    for _ in 0..40 { app.tick(&mut scene, 0.05); }
    assert_eq!(app.current_tick(), 0);
}
```

- [ ] **Step 2: Run test, expect compile failure (VizApp missing)**

Run: `cargo test -p viz --test tick_advances`
Expected: FAIL — `viz::app` module does not exist.

- [ ] **Step 3: Write `crates/viz/src/app.rs`**

```rust
//! `VizApp` — the `voxel_engine::app::App` implementation that drives the
//! engine's tick loop. Owns `SimState` + `SimScratch` + `EventRing` +
//! `CascadeRegistry`, plus a wall-clock accumulator that fires a fixed-step
//! `engine::step::step(...)` every `TICK_MS` of un-paused time.
//!
//! Rendering is intentionally out-of-scope for this file (see `render.rs`).
//! Input handling (keyboard) is also split out so the app module is easy to
//! unit-test without a window.

use crate::render::AgentRenderer;
use crate::scenario::Scenario;
use anyhow::Result;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::SimState;
use engine::step::{step, SimScratch};
use voxel_engine::app::App;
use voxel_engine::scene::Scene;
use winit::event::WindowEvent;

/// Target sim-tick period in seconds. Matches engine spec §2 (100 ms fixed
/// tick); real time / 0.1 s = sim ticks to run.
pub const TICK_SEC: f32 = 0.1;

/// Agent-cap headroom above the scenario spawn count; cheap to size up.
const AGENT_CAP_HEADROOM: u32 = 64;

pub struct VizApp {
    scenario:     Scenario,
    seed:         u64,
    state:        SimState,
    scratch:      SimScratch,
    events:       EventRing,
    cascade:      CascadeRegistry,
    backend:      UtilityBackend,
    accumulator:  f32,
    paused:       bool,
    speed:        f32,              // 1.0 = real time; halved by `[`, doubled by `]`
    renderer:     AgentRenderer,
    wall_seconds: f32,              // elapsed real seconds, for HUD
}

impl VizApp {
    /// Build a fresh VizApp from a loaded scenario + seed. Does NOT spawn
    /// agents yet — that happens in `setup()` once the Scene is available.
    pub fn from_scenario(scenario: Scenario, seed: u64) -> Self {
        let cap = scenario.agents.len() as u32 + AGENT_CAP_HEADROOM;
        let state = SimState::new(cap.max(1), seed);
        let scratch = SimScratch::new(state.agent_cap() as usize);
        let events = EventRing::with_cap(4096);
        Self {
            scenario,
            seed,
            state,
            scratch,
            events,
            cascade:      CascadeRegistry::new(),
            backend:      UtilityBackend,
            accumulator:  0.0,
            paused:       false,
            speed:        1.0,
            renderer:     AgentRenderer::new(),
            wall_seconds: 0.0,
        }
    }

    pub fn current_tick(&self) -> u32 { self.state.tick }
    pub fn alive_count(&self) -> usize { self.state.agents_alive().count() }
    pub fn paused(&self) -> bool { self.paused }
    pub fn speed(&self) -> f32 { self.speed }
    pub fn wall_seconds(&self) -> f32 { self.wall_seconds }

    pub fn set_paused(&mut self, p: bool) { self.paused = p; }
    pub fn toggle_paused(&mut self) { self.paused = !self.paused; }
    pub fn speed_up(&mut self)   { self.speed = (self.speed * 2.0).min(16.0); }
    pub fn speed_down(&mut self) { self.speed = (self.speed / 2.0).max(0.0625); }

    /// Single-step one engine tick (regardless of pause / accumulator).
    /// Returns the new tick number.
    pub fn step_one(&mut self, scene: &mut Scene) -> u32 {
        step(&mut self.state, &mut self.scratch, &mut self.events,
             &self.backend, &self.cascade);
        self.renderer.observe_events(&self.events, self.state.tick);
        self.renderer.redraw(&self.state, scene);
        self.state.tick
    }

    /// Wipe agents + events, reload scenario, reset tick counter.
    pub fn reset(&mut self, scene: &mut Scene) -> Result<()> {
        let cap = self.scenario.agents.len() as u32 + AGENT_CAP_HEADROOM;
        self.state = SimState::new(cap.max(1), self.seed);
        self.scratch = SimScratch::new(self.state.agent_cap() as usize);
        self.events  = EventRing::with_cap(4096);
        self.accumulator = 0.0;
        self.wall_seconds = 0.0;
        self.renderer.reset(scene);
        self.spawn_all_agents()?;
        self.renderer.redraw(&self.state, scene);
        Ok(())
    }

    fn spawn_all_agents(&mut self) -> Result<()> {
        for spec in &self.scenario.agents {
            let spawn = spec.to_engine();
            if self.state.spawn_agent(spawn).is_none() {
                anyhow::bail!("agent pool full (cap={})", self.state.agent_cap());
            }
        }
        Ok(())
    }
}

impl App for VizApp {
    fn setup(&mut self, scene: &mut Scene) -> Result<()> {
        self.spawn_all_agents()?;
        self.renderer.redraw(&self.state, scene);
        Ok(())
    }

    fn tick(&mut self, scene: &mut Scene, dt: f32) {
        self.wall_seconds += dt;
        if self.paused { return; }

        self.accumulator += dt * self.speed;
        let mut ticked = false;
        while self.accumulator >= TICK_SEC {
            self.accumulator -= TICK_SEC;
            step(&mut self.state, &mut self.scratch, &mut self.events,
                 &self.backend, &self.cascade);
            self.renderer.observe_events(&self.events, self.state.tick);
            ticked = true;
        }
        if ticked {
            self.renderer.redraw(&self.state, scene);
        }
    }

    fn on_input(&mut self, _scene: &mut Scene, _event: &WindowEvent) {
        // Keyboard handling lives in Task 6. Scaffolding kept minimal here so
        // Task 3's smoke test doesn't need a window-event factory.
    }
}
```

- [ ] **Step 4: Create renderer placeholder so `app.rs` compiles**

We need a placeholder `AgentRenderer` until Task 4. Create `crates/viz/src/render.rs`:

```rust
//! Agent + event overlay rendering. Task 4 fills in the marker-voxel logic;
//! this file is a compile-only skeleton so `VizApp` can hold a `renderer`
//! field from Task 3 onward.

use engine::event::EventRing;
use engine::state::SimState;
use voxel_engine::scene::Scene;

pub struct AgentRenderer {
    // Populated in Task 4.
}

impl AgentRenderer {
    pub fn new() -> Self { Self {} }
    pub fn reset(&mut self, _scene: &mut Scene) {}
    pub fn redraw(&mut self, _state: &SimState, _scene: &mut Scene) {}
    pub fn observe_events(&mut self, _events: &EventRing, _tick: u32) {}
}
```

- [ ] **Step 5: Wire modules in `lib.rs`**

Replace `crates/viz/src/lib.rs` contents:

```rust
//! Visualization harness library. `main.rs` depends on this crate's public
//! modules; tests import from the crate root (`viz::scenario::...`).

pub mod app;
pub mod render;
pub mod scenario;
```

- [ ] **Step 6: Rewrite `crates/viz/src/main.rs` to drive `VizApp` with a CLI arg**

```rust
//! Visualization harness entry point. Opens a winit window, constructs a
//! headless voxel_engine `Scene`, drives a `VizApp` via a hand-rolled winit
//! event loop (voxel_engine does not yet ship `run_app(cfg, app)` — see
//! Plan 3.1 follow-up).

use anyhow::{Context, Result};
use std::time::Instant;
use viz::app::VizApp;
use viz::scenario::load_scenario_path;
use voxel_engine::app::{App, AppConfig};
use voxel_engine::scene::{Scene, SceneConfig};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct Driver<A: App> {
    cfg:          AppConfig,
    app:          A,
    scene:        Scene,
    window:       Option<Window>,
    last_tick:    Instant,
    setup_called: bool,
}

impl<A: App> Driver<A> {
    fn new(cfg: AppConfig, app: A) -> Self {
        let scene = Scene::new_headless(SceneConfig::default());
        Self {
            cfg, app, scene,
            window: None,
            last_tick: Instant::now(),
            setup_called: false,
        }
    }
}

impl<A: App> ApplicationHandler for Driver<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title(self.cfg.window_title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.cfg.width as f64,
                self.cfg.height as f64,
            ));
        let w = event_loop.create_window(attrs).expect("create window");
        self.window = Some(w);
        if !self.setup_called {
            self.app.setup(&mut self.scene).expect("App::setup failed");
            self.setup_called = true;
            self.last_tick = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        self.app.on_input(&mut self.scene, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_tick).as_secs_f32();
                self.last_tick = now;
                self.app.tick(&mut self.scene, dt);
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let scenario_path = std::env::args().nth(1)
        .context("usage: viz <scenario.toml>")?;
    let scenario = load_scenario_path(&scenario_path)?;
    let seed: u64 = std::env::var("VIZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cfg = AppConfig {
        window_title: format!("World Sim Viz — {}", scenario_path),
        width: 1280, height: 720,
        ..Default::default()
    };
    let app = VizApp::from_scenario(scenario, seed);

    let event_loop = EventLoop::new()?;
    let mut driver = Driver::new(cfg, app);
    event_loop.run_app(&mut driver)?;
    Ok(())
}
```

- [ ] **Step 7: Run the smoke test, expect pass**

Run: `cargo test -p viz --test tick_advances`
Expected: all 3 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/viz/src/app.rs crates/viz/src/render.rs crates/viz/src/lib.rs crates/viz/src/main.rs crates/viz/tests/tick_advances.rs
git commit -m "feat(viz): VizApp drives engine::step at fixed 100ms cadence (Task 3/8)"
```

---

## Task 4: Agent rendering as colored voxel markers

**Files:**
- Modify: `crates/viz/src/render.rs` (replace placeholder with real impl)
- Create: `crates/viz/tests/agent_rendering.rs`

**Rendering model (explained once, referenced by later tasks):**
voxel_engine's `Scene` owns a list of *entities*, each of which carries its own
`VoxelGrid` + `MaterialPalette` and a world-space `Transform`. To show N agents
we spawn a **single "viz-layer" entity** the first time we redraw. Its grid is
sized large enough for the whole visible area; we translate world positions
into grid-cell coordinates (1 world-meter = 1 voxel cell). Each redraw:

1. Sweep the grid, zero any previously-set marker cells.
2. For each alive agent, compute `(cx, cy, cz)` from its position and call
   `scene.set_voxel(handle, IVec3::new(cx, cy, cz), color_index)`.
3. Overlay cells (from Task 5 events) are set the same way, on top.

Palette: index 0 = Air (forced), 1 = Human (white), 2 = Wolf (grey),
3 = Deer (tan), 4 = Dragon (red), 10 = Attack line (red), 11 = Death marker
(black), 12 = Announce ring (yellow).

World → grid mapping: `cx = (pos.x + GRID_RADIUS) as i32` etc., so world origin
lives at grid-cell `(GRID_RADIUS, GRID_RADIUS, GRID_RADIUS)`. Positions outside
±GRID_RADIUS are silently clipped. We DON'T try to auto-scroll the grid — MVP.

`scene.set_voxel(handle, pos, mat)` queues a mutation applied on the *next*
`scene.tick_sim()`. Since VizApp never calls `tick_sim` in MVP (renderer isn't
GPU-connected yet), the mutations accumulate harmlessly in the pending-mutation
queue. For correctness under tests we **call `scene.tick_sim()` at the end of
every `redraw`** so `voxel_count(handle)` reflects current state — this is a
no-op for cleanup since the viz-layer entity is never spawned as a fragment.

- [ ] **Step 1: Write failing tests**

```rust
// crates/viz/tests/agent_rendering.rs
use viz::app::VizApp;
use viz::render::{AgentRenderer, GRID_DIM, GRID_RADIUS, MAT_HUMAN, MAT_WOLF};
use viz::scenario::Scenario;
use voxel_engine::app::App;
use voxel_engine::scene::{Scene, SceneConfig};
use glam::IVec3;

fn basic_scen() -> Scenario {
    toml::from_str(r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0

        [[agent]]
        creature_type = "Wolf"
        pos = [3.0, 0.0, 4.0]
        hp = 100.0
    "#).unwrap()
}

#[test]
fn redraw_sets_one_voxel_per_agent() {
    let mut app = VizApp::from_scenario(basic_scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    let handle = app.debug_viz_entity().expect("viz entity spawned after setup");
    // Two agents → two non-empty voxels on the viz-layer entity.
    assert_eq!(scene.voxel_count(handle), Some(2));
}

#[test]
fn color_depends_on_creature_type() {
    let mut app = VizApp::from_scenario(basic_scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    let handle = app.debug_viz_entity().unwrap();

    // Human at world (0,0,0) → grid (GRID_RADIUS, GRID_RADIUS, GRID_RADIUS).
    let human_cell = IVec3::splat(GRID_RADIUS as i32);
    assert_eq!(
        AgentRenderer::debug_get_voxel(&scene, handle, human_cell),
        Some(MAT_HUMAN),
    );

    // Wolf at (3, 0, 4) → (GRID_RADIUS+3, GRID_RADIUS, GRID_RADIUS+4).
    let wolf_cell = IVec3::new(
        GRID_RADIUS as i32 + 3,
        GRID_RADIUS as i32,
        GRID_RADIUS as i32 + 4,
    );
    assert_eq!(
        AgentRenderer::debug_get_voxel(&scene, handle, wolf_cell),
        Some(MAT_WOLF),
    );
}

#[test]
fn grid_dim_is_power_of_two_and_large_enough() {
    assert!(GRID_DIM >= 128);
    assert_eq!(GRID_DIM % 2, 0);
    assert_eq!(GRID_RADIUS * 2, GRID_DIM);
}
```

- [ ] **Step 2: Run test, expect compile failure**

Run: `cargo test -p viz --test agent_rendering`
Expected: FAIL — `GRID_DIM`, `debug_viz_entity`, etc. do not exist.

- [ ] **Step 3: Replace `crates/viz/src/render.rs` with the real impl**

```rust
//! Agent + event overlay rendering. Agents are single-voxel markers on a
//! dedicated "viz-layer" scene entity; colors are palette-indexed by creature
//! type. Event overlays (Task 5) reuse the same grid.
//!
//! NOTE: voxel_engine's rendering path (VoxelRenderer / VulkanContext) is not
//! wired into the App driver yet — the window stays blank. This module still
//! writes the correct voxel data so (a) tests can assert on it and (b) when
//! Plan 3.1 lands a renderer, agents appear with zero code changes here.

use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::SimState;
use glam::{IVec3, Vec3};
use std::collections::VecDeque;
use voxel_engine::scene::{EntityHandle, Scene, Transform};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};

/// Grid cube side length, in voxel cells (= world meters). Must be even.
pub const GRID_DIM:    u32 = 256;
/// Half of `GRID_DIM`; world origin maps to `(GRID_RADIUS, GRID_RADIUS, GRID_RADIUS)`.
pub const GRID_RADIUS: u32 = GRID_DIM / 2;

// Palette indices. 0 is reserved (Air).
pub const MAT_HUMAN:      u8 = 1;
pub const MAT_WOLF:       u8 = 2;
pub const MAT_DEER:       u8 = 3;
pub const MAT_DRAGON:     u8 = 4;
pub const MAT_ATTACK:     u8 = 10;  // red line cell
pub const MAT_DEATH:      u8 = 11;  // black death marker
pub const MAT_ANNOUNCE:   u8 = 12;  // yellow announce ring
// ...room for more without colliding with agent colors.

/// How many ticks a transient overlay cell (attack line, announce ring) lingers.
pub const OVERLAY_TTL_TICKS:     u32 = 5;
/// Announce ring expands 1m → 3m → 5m over 3 ticks.
pub const ANNOUNCE_RING_RADII_M: [i32; 3] = [1, 3, 5];

/// An overlay voxel with a lifetime in sim ticks. When `expires_at_tick`
/// equals the current tick, the cell is cleared on next redraw.
#[derive(Copy, Clone, Debug)]
struct OverlayCell {
    cell:             IVec3,
    mat:              u8,
    expires_at_tick:  u32,
    /// If true, the cell is permanent (death markers). `expires_at_tick` ignored.
    permanent:        bool,
}

pub struct AgentRenderer {
    /// The sole scene entity all voxel writes go into. Allocated lazily on
    /// first `redraw` (we need the `Scene` to spawn it).
    viz_entity:     Option<EntityHandle>,
    /// Cells set by the LAST `redraw` for agents (cleared at top of next redraw).
    prev_agents:    Vec<IVec3>,
    /// Event-driven overlays (attack lines, death markers, announce rings).
    overlays:       VecDeque<OverlayCell>,
    /// Highest sim-tick we've already processed in `observe_events` — lets us
    /// avoid re-processing the same EventRing entries across redraws.
    last_obs_tick:  u32,
}

impl AgentRenderer {
    pub fn new() -> Self {
        Self {
            viz_entity:    None,
            prev_agents:   Vec::new(),
            overlays:      VecDeque::new(),
            last_obs_tick: 0,
        }
    }

    /// Despawn the viz entity so the next `redraw` reallocates a clean one.
    pub fn reset(&mut self, scene: &mut Scene) {
        if let Some(h) = self.viz_entity.take() {
            scene.despawn(h);
        }
        self.prev_agents.clear();
        self.overlays.clear();
        self.last_obs_tick = 0;
    }

    /// Drain new events since the previous call and translate them into
    /// overlay cells. Called immediately after `engine::step::step` returns,
    /// BEFORE `redraw` (so overlays are visible the same tick they happen).
    pub fn observe_events(&mut self, events: &EventRing, current_tick: u32) {
        // EventRing holds at most `cap` entries; we walk them in order and
        // only consume ones whose `tick` > `last_obs_tick`.
        let mut agent_positions: Vec<(AgentId, Vec3)> = Vec::new();
        // Build a position lookup lazily — we need it for attack-line endpoints,
        // but we don't have a &SimState here. We'll rebuild by re-scanning the
        // event ring for the most-recent `AgentMoved` before the attack event.
        // For MVP: attack lines use the snapshot of attacker/target pos at
        // the moment the event was observed, approximated by the last-known
        // position from recent AgentMoved events. If no move has happened yet,
        // we skip drawing the line. See comment in `attack_line_endpoints`.
        let _ = &mut agent_positions;

        for event in events.iter() {
            let evt_tick = event.tick();
            if evt_tick <= self.last_obs_tick { continue; }
            match event {
                Event::AgentAttacked { attacker, target, .. } => {
                    // We need positions; see `redraw` path — we stash the
                    // attack as a "pending" overlay keyed on AgentIds, then
                    // resolve to cells when `redraw` runs with the SimState.
                    self.overlays.push_back(OverlayCell {
                        cell:            agent_pair_marker(*attacker, *target),
                        mat:             MAT_ATTACK,
                        expires_at_tick: current_tick + OVERLAY_TTL_TICKS,
                        permanent:       false,
                    });
                }
                Event::AgentDied { agent_id, .. } => {
                    // Resolved to a real cell in `redraw` via
                    // `agent_cell_or_marker` using the last-known position.
                    self.overlays.push_back(OverlayCell {
                        cell:            agent_id_marker(*agent_id),
                        mat:             MAT_DEATH,
                        expires_at_tick: 0,
                        permanent:       true,
                    });
                }
                Event::AnnounceEmitted { speaker, .. } => {
                    self.overlays.push_back(OverlayCell {
                        cell:            agent_id_marker(*speaker),
                        mat:             MAT_ANNOUNCE,
                        expires_at_tick: current_tick + 3,
                        permanent:       false,
                    });
                }
                _ => {}
            }
        }
        self.last_obs_tick = current_tick;
    }

    pub fn redraw(&mut self, state: &SimState, scene: &mut Scene) {
        // 1. Lazily allocate the viz entity.
        let handle = match self.viz_entity {
            Some(h) => h,
            None => {
                let h = spawn_viz_entity(scene);
                self.viz_entity = Some(h);
                h
            }
        };

        // 2. Clear previous-frame agent cells.
        for cell in self.prev_agents.drain(..) {
            scene.set_voxel(handle, cell, 0);
        }

        // 3. Rewrite agent markers.
        for agent in state.agents_alive() {
            let pos    = match state.agent_pos(agent) { Some(p) => p, None => continue };
            let ctype  = state.agent_creature_type(agent).unwrap_or(CreatureType::Human);
            let cell   = world_to_cell(pos);
            if !in_grid(cell) { continue; }
            let mat    = creature_color(ctype);
            scene.set_voxel(handle, cell, mat);
            self.prev_agents.push(cell);
        }

        // 4. Re-apply persistent overlays; expire transient ones.
        let tick = state.tick;
        let mut i = 0;
        while i < self.overlays.len() {
            let o = self.overlays[i];
            let expired = !o.permanent && tick >= o.expires_at_tick;
            if expired {
                scene.set_voxel(handle, o.cell, 0);
                self.overlays.remove(i);
            } else {
                // `cell` was encoded from agent IDs in `observe_events`; resolve
                // now that we have state. If the marker is an `agent_id_marker`
                // sentinel, promote to the agent's current voxel cell.
                let real_cell = resolve_marker_cell(o.cell, state).unwrap_or(o.cell);
                scene.set_voxel(handle, real_cell, o.mat);
                if real_cell != o.cell {
                    self.overlays[i].cell = real_cell;
                }
                i += 1;
            }
        }

        // 5. Flush pending mutations so tests can observe via `voxel_count`.
        scene.tick_sim();
    }

    /// Accessor for tests: fetch a voxel value by entity handle and cell.
    /// Uses a single-voxel scan via `scene.voxel_count` → not available; we
    /// read through `Scene`'s public API by... there is none that returns
    /// arbitrary voxel values. So we record test-observable state ourselves.
    pub fn debug_get_voxel(scene: &Scene, handle: EntityHandle, cell: IVec3)
        -> Option<u8>
    {
        // voxel_engine's Scene does not currently expose `get_voxel`. We
        // approximate by round-tripping through `voxel_count` deltas: if
        // setting the cell to Air drops the count, the cell was non-empty.
        // For precise value testing we would need a voxel_engine API addition
        // (flagged in Plan 3.1). For MVP we return `Some(1)` if the cell is
        // set to any non-zero value, `Some(0)` if empty, `None` if out-of-grid.
        //
        // Test note: `agent_rendering.rs::color_depends_on_creature_type`
        // requires distinguishing MAT_HUMAN from MAT_WOLF. We satisfy this
        // by queuing a speculative `set_voxel(..., test_material)` and
        // comparing `voxel_count` before / after — this is ugly but avoids
        // an API change. See `voxel_engine_gap` note in this plan's header.
        let _ = (scene, handle, cell);
        None
    }
}

fn spawn_viz_entity(scene: &mut Scene) -> EntityHandle {
    let grid = VoxelGrid::new(GRID_DIM, GRID_DIM, GRID_DIM);
    let palette = viz_palette();
    let tf = Transform::default();
    scene.spawn(&grid, tf, &palette)
}

fn viz_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    p.set(MAT_HUMAN,    PaletteEntry { r: 240, g: 240, b: 240, roughness: 128, emissive: 0, material_type: MaterialType::Plastic });
    p.set(MAT_WOLF,     PaletteEntry { r: 120, g: 120, b: 120, roughness: 128, emissive: 0, material_type: MaterialType::Plastic });
    p.set(MAT_DEER,     PaletteEntry { r: 180, g: 140, b:  80, roughness: 128, emissive: 0, material_type: MaterialType::Plastic });
    p.set(MAT_DRAGON,   PaletteEntry { r: 200, g:  40, b:  40, roughness: 128, emissive: 0, material_type: MaterialType::Plastic });
    p.set(MAT_ATTACK,   PaletteEntry { r: 255, g:   0, b:   0, roughness: 128, emissive: 200, material_type: MaterialType::Plastic });
    p.set(MAT_DEATH,    PaletteEntry { r:  10, g:  10, b:  10, roughness: 128, emissive: 0, material_type: MaterialType::Plastic });
    p.set(MAT_ANNOUNCE, PaletteEntry { r: 230, g: 210, b:  40, roughness: 128, emissive: 200, material_type: MaterialType::Plastic });
    p
}

fn creature_color(ct: CreatureType) -> u8 {
    match ct {
        CreatureType::Human  => MAT_HUMAN,
        CreatureType::Wolf   => MAT_WOLF,
        CreatureType::Deer   => MAT_DEER,
        CreatureType::Dragon => MAT_DRAGON,
    }
}

fn world_to_cell(pos: Vec3) -> IVec3 {
    IVec3::new(
        (pos.x + GRID_RADIUS as f32).round() as i32,
        (pos.y + GRID_RADIUS as f32).round() as i32,
        (pos.z + GRID_RADIUS as f32).round() as i32,
    )
}

fn in_grid(c: IVec3) -> bool {
    c.x >= 0 && c.x < GRID_DIM as i32 &&
    c.y >= 0 && c.y < GRID_DIM as i32 &&
    c.z >= 0 && c.z < GRID_DIM as i32
}

// ---- Marker sentinels -------------------------------------------------------
//
// `observe_events` runs BEFORE `redraw`, so we don't have positions to resolve
// attack endpoints / death locations. We encode the agent ID as a sentinel
// `IVec3` (`x = AgentId::raw(), y = i32::MIN, z = i32::MIN` for solo-agent
// markers; `x = attacker, y = i32::MIN, z = target` for pair markers) and
// resolve to a real cell in `redraw`. Sentinel cells never collide with real
// grid cells because `y = i32::MIN` is out of grid bounds.

fn agent_id_marker(id: AgentId) -> IVec3 {
    IVec3::new(id.raw() as i32, i32::MIN, i32::MIN)
}

fn agent_pair_marker(a: AgentId, b: AgentId) -> IVec3 {
    IVec3::new(a.raw() as i32, i32::MIN, b.raw() as i32)
}

fn is_solo_marker(c: IVec3) -> bool { c.y == i32::MIN && c.z == i32::MIN }
fn is_pair_marker(c: IVec3) -> bool { c.y == i32::MIN && c.z != i32::MIN }

fn resolve_marker_cell(c: IVec3, state: &SimState) -> Option<IVec3> {
    if is_pair_marker(c) {
        let a = AgentId::new(c.x as u32)?;
        let b = AgentId::new(c.z as u32)?;
        let pa = state.agent_pos(a)?;
        let pb = state.agent_pos(b)?;
        // Midpoint for simplicity.
        let mid = (pa + pb) * 0.5;
        Some(world_to_cell(mid))
    } else if is_solo_marker(c) {
        let a = AgentId::new(c.x as u32)?;
        // Fallback: if the agent is dead, just keep the sentinel (but return
        // None so the caller skips drawing this frame). Death markers should
        // capture the last-known pos — that requires storing on event observe;
        // for MVP we live with "death marker tracks last-known body position".
        let p = state.agent_pos(a)?;
        Some(world_to_cell(p))
    } else {
        None
    }
}
```

- [ ] **Step 4: Make `AgentRenderer::debug_get_voxel` actually work for tests**

The stub above admits it can't distinguish colors. voxel_engine's `Scene` doesn't expose `get_voxel` on entity grids. **Rather than blocking on a voxel_engine API change, add a test-only parallel shadow buffer inside `AgentRenderer`.** Replace the `debug_get_voxel` stub with:

```rust
    /// Test accessor: return the material index most recently written to
    /// `cell` by this renderer (0 if cleared / never written). Tracks writes
    /// through a shadow buffer keyed by cell; voxel_engine's `Scene` does not
    /// expose a `get_voxel` API as of 2026-04-19 (see Plan 3.1 follow-up).
    pub fn debug_get_voxel(scene: &Scene, handle: EntityHandle, cell: IVec3)
        -> Option<u8>
    {
        // Accessed via the per-renderer shadow; the caller uses
        // `VizApp::debug_get_voxel(cell)` instead. This free function is
        // kept for API symmetry but simply defers.
        let _ = (scene, handle, cell);
        None
    }
```

And add a shadow buffer + accessor on `AgentRenderer`:

```rust
// ... inside `pub struct AgentRenderer { ... }`
    /// Shadow of the last voxel value written per cell. Test-only. Cleared
    /// cells map to 0.
    shadow: ahash::AHashMap<(i32, i32, i32), u8>,
```

Update `AgentRenderer::new`:

```rust
    pub fn new() -> Self {
        Self {
            viz_entity:    None,
            prev_agents:   Vec::new(),
            overlays:      VecDeque::new(),
            last_obs_tick: 0,
            shadow:        ahash::AHashMap::new(),
        }
    }
```

Wrap every `scene.set_voxel(...)` call in a small helper that updates the shadow:

```rust
fn write_voxel(
    renderer: &mut AgentRenderer,
    scene:    &mut Scene,
    handle:   EntityHandle,
    cell:     IVec3,
    mat:      u8,
) {
    scene.set_voxel(handle, cell, mat);
    let k = (cell.x, cell.y, cell.z);
    if mat == 0 { renderer.shadow.remove(&k); }
    else        { renderer.shadow.insert(k, mat); }
}
```

Replace every `scene.set_voxel(...)` call inside `AgentRenderer::redraw` with `write_voxel(self, scene, handle, ...)`. Add the shadow accessor:

```rust
impl AgentRenderer {
    pub fn shadow_get(&self, cell: IVec3) -> u8 {
        self.shadow.get(&(cell.x, cell.y, cell.z)).copied().unwrap_or(0)
    }
}
```

Add `ahash` to `crates/viz/Cargo.toml` dependencies:

```toml
ahash = "0.8"
```

- [ ] **Step 5: Expose renderer access on VizApp for tests**

Modify `crates/viz/src/app.rs` — add an accessor:

```rust
impl VizApp {
    // ...
    pub fn debug_viz_entity(&self) -> Option<voxel_engine::scene::EntityHandle> {
        self.renderer.debug_viz_entity()
    }
    pub fn debug_shadow_voxel(&self, cell: glam::IVec3) -> u8 {
        self.renderer.shadow_get(cell)
    }
}
```

And on `AgentRenderer`:

```rust
impl AgentRenderer {
    pub fn debug_viz_entity(&self) -> Option<EntityHandle> { self.viz_entity }
}
```

- [ ] **Step 6: Rewrite the color test to use the shadow buffer**

Replace the body of `color_depends_on_creature_type` in `crates/viz/tests/agent_rendering.rs`:

```rust
#[test]
fn color_depends_on_creature_type() {
    let mut app = VizApp::from_scenario(basic_scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    let human_cell = IVec3::splat(GRID_RADIUS as i32);
    assert_eq!(app.debug_shadow_voxel(human_cell), MAT_HUMAN);

    let wolf_cell = IVec3::new(
        GRID_RADIUS as i32 + 3,
        GRID_RADIUS as i32,
        GRID_RADIUS as i32 + 4,
    );
    assert_eq!(app.debug_shadow_voxel(wolf_cell), MAT_WOLF);
}
```

- [ ] **Step 7: Run tests, expect pass**

Run: `cargo test -p viz --test agent_rendering`
Expected: all 3 tests PASS.

Run: `cargo test -p viz` (full suite)
Expected: all tests PASS (Task 2 + Task 3 + Task 4 suites).

- [ ] **Step 8: Commit**

```bash
git add crates/viz/Cargo.toml crates/viz/src/render.rs crates/viz/src/app.rs crates/viz/tests/agent_rendering.rs
git commit -m "feat(viz): agent markers as colored voxels on a dedicated scene entity (Task 4/8)"
```

---

## Task 5: Event overlays — Attack / Death / Announce

**Files:**
- Modify: `crates/viz/src/render.rs` (flesh out `observe_events` and overlay lifecycle; Task 4 already stubbed the entry points)
- Create: `crates/viz/tests/event_overlays.rs`

Most of the overlay lifecycle is already wired from Task 4's skeleton. This task completes the **announce ring expansion** (single marker → 1m → 3m → 5m over 3 ticks) and adds regression tests.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/viz/tests/event_overlays.rs
use viz::app::VizApp;
use viz::render::{GRID_DIM, GRID_RADIUS, MAT_ATTACK, MAT_DEATH, MAT_ANNOUNCE, OVERLAY_TTL_TICKS};
use viz::scenario::Scenario;
use voxel_engine::app::App;
use voxel_engine::scene::{Scene, SceneConfig};
use glam::IVec3;

fn two_agents_1m_apart() -> Scenario {
    // Two humans ~1 m apart → UtilityBackend's Attack score wins once in range.
    toml::from_str(r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0

        [[agent]]
        creature_type = "Human"
        pos = [1.5, 0.0, 0.0]
        hp = 20.0
    "#).unwrap()
}

#[test]
fn attack_creates_red_overlay_that_expires_after_ttl() {
    let mut app = VizApp::from_scenario(two_agents_1m_apart(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    // Step until we observe an attack-overlay set. UtilityBackend attacks
    // when in range; distance 1.5 m < ATTACK_RANGE=2.0.
    let mut saw_red = false;
    for _ in 0..20 {
        app.step_one(&mut scene);
        // Midpoint of the two agents is ~(0.75, 0, 0) → cell ~(GR, GR, GR) offset by 1.
        let mid = IVec3::new(GRID_RADIUS as i32 + 1, GRID_RADIUS as i32, GRID_RADIUS as i32);
        if app.debug_shadow_voxel(mid) == MAT_ATTACK {
            saw_red = true;
            break;
        }
    }
    assert!(saw_red, "expected at least one MAT_ATTACK overlay cell");

    // Run enough further ticks that all attack overlays expire.
    for _ in 0..(OVERLAY_TTL_TICKS + 2) {
        app.step_one(&mut scene);
    }
    // No attack overlays should remain on the midpoint (count may vary as
    // agents move; assert on the specific cell).
    let mid = IVec3::new(GRID_RADIUS as i32 + 1, GRID_RADIUS as i32, GRID_RADIUS as i32);
    assert_ne!(app.debug_shadow_voxel(mid), MAT_ATTACK);
}

#[test]
fn death_overlay_is_permanent() {
    let mut app = VizApp::from_scenario(two_agents_1m_apart(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    // Run 200 ticks — HP 20 target dies in at most 2 hits (10 dmg each).
    for _ in 0..200 { app.step_one(&mut scene); }

    // Somewhere in the grid we expect a DEATH marker; exact cell depends on
    // where the corpse ended up. Count occurrences in the shadow.
    let mut death_cells = 0;
    for x in 0..GRID_DIM as i32 {
        for z in 0..GRID_DIM as i32 {
            if app.debug_shadow_voxel(IVec3::new(x, GRID_RADIUS as i32, z)) == MAT_DEATH {
                death_cells += 1;
            }
        }
    }
    assert!(death_cells >= 1, "expected at least one MAT_DEATH marker, got {}", death_cells);
}

#[test]
fn announce_ring_appears_at_speaker_position() {
    // We can't directly emit an Announce from UtilityBackend (it only picks
    // micros). Instead, push a synthetic Announce event into the ring via
    // the public API and observe the renderer's response.
    use engine::event::{Event, EventRing};
    use engine::ids::AgentId;

    let scen: Scenario = toml::from_str(r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0
    "#).unwrap();
    let mut app = VizApp::from_scenario(scen, 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();

    // Directly exercise the renderer via a throwaway event ring; this is a
    // white-box test, not a full sim run.
    let mut ring = EventRing::with_cap(16);
    ring.push(Event::AnnounceEmitted {
        speaker:      AgentId::new(1).unwrap(),
        audience_tag: 2, // Anyone
        fact_payload: 0,
        tick:         1,
    });
    app.debug_observe_events(&ring, 1);
    app.debug_redraw(&mut scene);

    let speaker_cell = IVec3::splat(GRID_RADIUS as i32);
    assert_eq!(app.debug_shadow_voxel(speaker_cell), MAT_ANNOUNCE);
}
```

- [ ] **Step 2: Run tests, expect compile failures**

Run: `cargo test -p viz --test event_overlays`
Expected: FAIL — `debug_observe_events`, `debug_redraw` missing on `VizApp`.

- [ ] **Step 3: Add debug helpers on `VizApp`**

Modify `crates/viz/src/app.rs`:

```rust
impl VizApp {
    // ... existing methods ...

    /// Test helper: directly feed events into the renderer's observe stage.
    pub fn debug_observe_events(&mut self, events: &engine::event::EventRing, tick: u32) {
        self.renderer.observe_events(events, tick);
    }

    /// Test helper: force a redraw against an arbitrary scene.
    pub fn debug_redraw(&mut self, scene: &mut Scene) {
        self.renderer.redraw(&self.state, scene);
    }
}
```

- [ ] **Step 4: Run tests, expect PASS on most, partial on announce ring**

Run: `cargo test -p viz --test event_overlays`
Expected: `attack_creates_red_overlay_that_expires_after_ttl` and `death_overlay_is_permanent` and `announce_ring_appears_at_speaker_position` all PASS. The announce ring currently only places a single voxel at the speaker (not an expanding ring) — the test asserts the single voxel appears, which is satisfied. Ring expansion is deferred to Step 5.

- [ ] **Step 5: Add expanding-ring lifecycle for Announce**

Modify `observe_events` in `crates/viz/src/render.rs` — replace the `Event::AnnounceEmitted` arm:

```rust
                Event::AnnounceEmitted { speaker, .. } => {
                    // Emit three overlays, each with a larger radius marker
                    // and a staggered expiry. Because we don't have state here
                    // to read positions, we use the solo-marker sentinel and
                    // let `redraw` resolve it. The "ring" is approximated by
                    // three marker cells arranged in a plus-pattern around
                    // the speaker at radii 1, 3, 5 (see `emit_ring`).
                    for (offset_ticks, radius_m) in ANNOUNCE_RING_RADII_M.iter().enumerate() {
                        self.overlays.push_back(OverlayCell {
                            cell:            announce_ring_marker(*speaker, *radius_m),
                            mat:             MAT_ANNOUNCE,
                            expires_at_tick: current_tick + OVERLAY_TTL_TICKS + offset_ticks as u32,
                            permanent:       false,
                        });
                    }
                }
```

Add the sentinel + resolver. Near the other marker helpers:

```rust
fn announce_ring_marker(speaker: AgentId, radius_m: i32) -> IVec3 {
    // Encode (speaker, radius) into a unique sentinel. radius lives in `z`.
    IVec3::new(speaker.raw() as i32, i32::MIN, -radius_m)
}

fn is_announce_ring_marker(c: IVec3) -> bool {
    c.y == i32::MIN && c.z <= 0 && c.z >= -8
}
```

Extend `resolve_marker_cell`:

```rust
fn resolve_marker_cell(c: IVec3, state: &SimState) -> Option<IVec3> {
    if is_announce_ring_marker(c) {
        let a = AgentId::new(c.x as u32)?;
        let center = state.agent_pos(a)?;
        let r = (-c.z) as f32;
        // Pick one of the four cardinal-plus cells at radius r. Use c.z sign
        // bit... actually we emit 3 markers with the same (speaker, radius)
        // which collide. For an actual RING we'd emit 8 markers per radius.
        // MVP: place a single marker `r` units north of the speaker.
        let p = center + Vec3::new(0.0, 0.0, r);
        Some(world_to_cell(p))
    } else if is_pair_marker(c) {
        let a = AgentId::new(c.x as u32)?;
        let b = AgentId::new(c.z as u32)?;
        let pa = state.agent_pos(a)?;
        let pb = state.agent_pos(b)?;
        Some(world_to_cell((pa + pb) * 0.5))
    } else if is_solo_marker(c) {
        let a = AgentId::new(c.x as u32)?;
        let p = state.agent_pos(a)?;
        Some(world_to_cell(p))
    } else {
        None
    }
}
```

(Note the sentinel-discriminator ordering: announce-ring markers have `z <= 0`, pair markers have `z > 0` — the ordering in `resolve_marker_cell` matters.)

- [ ] **Step 6: Run tests, expect PASS**

Run: `cargo test -p viz`
Expected: all suites PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/viz/src/render.rs crates/viz/src/app.rs crates/viz/tests/event_overlays.rs
git commit -m "feat(viz): event overlays for Attack, Death, Announce with TTL ring buffer (Task 5/8)"
```

---

## Task 6: Keyboard controls + stdout HUD

**Files:**
- Modify: `crates/viz/src/app.rs` (populate `on_input`)
- Create: `crates/viz/src/hud.rs`
- Modify: `crates/viz/src/lib.rs` (`pub mod hud;`)
- Modify: `crates/viz/src/main.rs` (tick the HUD once per second)
- Create: `crates/viz/tests/keyboard.rs`

- [ ] **Step 1: Write failing keyboard test**

```rust
// crates/viz/tests/keyboard.rs
//
// Unit-test keyboard handling without a real window. We construct
// synthetic WindowEvent::KeyboardInput payloads and pass them to
// `VizApp::on_input` directly.

use viz::app::VizApp;
use viz::scenario::Scenario;
use voxel_engine::app::App;
use voxel_engine::scene::{Scene, SceneConfig};
use winit::event::{DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey, SmolStr};

fn scen() -> Scenario {
    toml::from_str(r#"
        [[agent]]
        creature_type = "Human"
        pos = [0.0, 0.0, 0.0]
        hp = 100.0
    "#).unwrap()
}

fn key_press(code: KeyCode) -> WindowEvent {
    WindowEvent::KeyboardInput {
        device_id: unsafe { DeviceId::dummy() },
        event: KeyEvent {
            physical_key: PhysicalKey::Code(code),
            logical_key:  Key::Named(NamedKey::Space),
            text:         None,
            location:     winit::keyboard::KeyLocation::Standard,
            state:        ElementState::Pressed,
            repeat:       false,
            platform_specific: Default::default(),
        },
        is_synthetic: false,
    }
}

#[test]
fn space_toggles_pause() {
    let mut app = VizApp::from_scenario(scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    assert!(!app.paused());
    app.on_input(&mut scene, &key_press(KeyCode::Space));
    assert!(app.paused());
    app.on_input(&mut scene, &key_press(KeyCode::Space));
    assert!(!app.paused());
}

#[test]
fn period_steps_one_tick_while_paused() {
    let mut app = VizApp::from_scenario(scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    app.on_input(&mut scene, &key_press(KeyCode::Space)); // pause
    assert_eq!(app.current_tick(), 0);
    app.on_input(&mut scene, &key_press(KeyCode::Period));
    assert_eq!(app.current_tick(), 1);
    app.on_input(&mut scene, &key_press(KeyCode::Period));
    assert_eq!(app.current_tick(), 2);
}

#[test]
fn r_resets_tick_to_zero() {
    let mut app = VizApp::from_scenario(scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    for _ in 0..10 { app.step_one(&mut scene); }
    assert_eq!(app.current_tick(), 10);
    app.on_input(&mut scene, &key_press(KeyCode::KeyR));
    assert_eq!(app.current_tick(), 0);
}

#[test]
fn bracket_keys_halve_and_double_speed() {
    let mut app = VizApp::from_scenario(scen(), 42);
    let mut scene = Scene::new_headless(SceneConfig::default());
    app.setup(&mut scene).unwrap();
    assert_eq!(app.speed(), 1.0);
    app.on_input(&mut scene, &key_press(KeyCode::BracketLeft));
    assert_eq!(app.speed(), 0.5);
    app.on_input(&mut scene, &key_press(KeyCode::BracketRight));
    app.on_input(&mut scene, &key_press(KeyCode::BracketRight));
    assert_eq!(app.speed(), 2.0);
}
```

- [ ] **Step 2: Run tests, expect failure**

Run: `cargo test -p viz --test keyboard`
Expected: FAIL — `on_input` does nothing yet.

- [ ] **Step 3: Implement `on_input` in `crates/viz/src/app.rs`**

Replace the existing `on_input` method:

```rust
    fn on_input(&mut self, scene: &mut Scene, event: &WindowEvent) {
        use winit::event::{ElementState, KeyEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        let WindowEvent::KeyboardInput { event: KeyEvent {
            state: ElementState::Pressed, physical_key: PhysicalKey::Code(code), ..
        }, .. } = event else { return; };

        match code {
            KeyCode::Space        => self.toggle_paused(),
            KeyCode::Period       => { let _ = self.step_one(scene); }
            KeyCode::KeyR         => { let _ = self.reset(scene); }
            KeyCode::BracketLeft  => self.speed_down(),
            KeyCode::BracketRight => self.speed_up(),
            _ => {}
        }
    }
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `cargo test -p viz --test keyboard`
Expected: all 4 tests PASS.

- [ ] **Step 5: Write the stdout HUD module**

Create `crates/viz/src/hud.rs`:

```rust
//! Stdout HUD — one line per second with tick counter + alive count + speed.
//! Minimal fallback because voxel_engine does not yet expose a text overlay
//! in the `App` trait surface we're targeting. Plan 3.1 can promote this to
//! an egui panel when the renderer is wired in.

use crate::app::VizApp;

pub struct Hud {
    last_emit_sec: f32,
}

impl Hud {
    pub fn new() -> Self { Self { last_emit_sec: 0.0 } }

    /// Emit one HUD line if at least 1 s has elapsed since the last emit.
    /// Also emits immediately on first call (so the user sees something).
    pub fn tick(&mut self, app: &VizApp) {
        let now = app.wall_seconds();
        if now - self.last_emit_sec < 1.0 && self.last_emit_sec != 0.0 {
            return;
        }
        self.last_emit_sec = if now == 0.0 { 0.001 } else { now };
        let paused = if app.paused() { " [PAUSED]" } else { "" };
        println!(
            "[viz] tick={:>6} alive={:>3} speed={:.2}x wall={:>6.1}s{}",
            app.current_tick(),
            app.alive_count(),
            app.speed(),
            app.wall_seconds(),
            paused,
        );
    }
}
```

- [ ] **Step 6: Wire the HUD into `main.rs`**

Modify `crates/viz/src/main.rs` — add `hud: Hud` to `Driver` and tick it each frame. Add the field:

```rust
use viz::hud::Hud;

struct Driver<A: App> {
    cfg:          AppConfig,
    app:          A,
    scene:        Scene,
    window:       Option<Window>,
    last_tick:    Instant,
    setup_called: bool,
    hud:          Hud,
}
```

Update `Driver::new`:

```rust
    fn new(cfg: AppConfig, app: A) -> Self {
        let scene = Scene::new_headless(SceneConfig::default());
        Self {
            cfg, app, scene,
            window:       None,
            last_tick:    Instant::now(),
            setup_called: false,
            hud:          Hud::new(),
        }
    }
```

**Concrete typing problem:** the HUD needs `&VizApp`, but `Driver<A: App>` is generic. Solve by making the driver monomorphic to `VizApp` — we don't actually need the generic:

Replace `struct Driver<A: App>` with `struct Driver` specialized to `VizApp`:

```rust
struct Driver {
    cfg:          AppConfig,
    app:          VizApp,
    scene:        Scene,
    window:       Option<Window>,
    last_tick:    Instant,
    setup_called: bool,
    hud:          Hud,
}

impl Driver {
    fn new(cfg: AppConfig, app: VizApp) -> Self {
        let scene = Scene::new_headless(SceneConfig::default());
        Self {
            cfg, app, scene,
            window:       None,
            last_tick:    Instant::now(),
            setup_called: false,
            hud:          Hud::new(),
        }
    }
}

impl ApplicationHandler for Driver {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title(self.cfg.window_title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.cfg.width as f64,
                self.cfg.height as f64,
            ));
        let w = event_loop.create_window(attrs).expect("create window");
        self.window = Some(w);
        if !self.setup_called {
            self.app.setup(&mut self.scene).expect("App::setup failed");
            self.setup_called = true;
            self.last_tick = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        self.app.on_input(&mut self.scene, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_tick).as_secs_f32();
                self.last_tick = now;
                self.app.tick(&mut self.scene, dt);
                self.hud.tick(&self.app);
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}
```

Update `main()`:

```rust
fn main() -> Result<()> {
    let scenario_path = std::env::args().nth(1)
        .context("usage: viz <scenario.toml>")?;
    let scenario = load_scenario_path(&scenario_path)?;
    let seed: u64 = std::env::var("VIZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cfg = AppConfig {
        window_title: format!("World Sim Viz — {}", scenario_path),
        width: 1280, height: 720,
        ..Default::default()
    };
    let app = VizApp::from_scenario(scenario, seed);

    let event_loop = EventLoop::new()?;
    let mut driver = Driver::new(cfg, app);
    event_loop.run_app(&mut driver)?;
    Ok(())
}
```

Update `crates/viz/src/lib.rs`:

```rust
pub mod app;
pub mod hud;
pub mod render;
pub mod scenario;
```

- [ ] **Step 7: Build + run full suite**

Run: `cargo build -p viz`
Expected: success.

Run: `cargo test -p viz`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/viz/src/app.rs crates/viz/src/hud.rs crates/viz/src/lib.rs crates/viz/src/main.rs crates/viz/tests/keyboard.rs
git commit -m "feat(viz): keyboard controls (pause/step/reset/speed) + stdout HUD (Task 6/8)"
```

---

## Task 7: `viz_basic.toml` scenario — 4 humans + 1 wolf

**Files:**
- Create: `crates/viz/scenarios/viz_basic.toml`
- Create: `crates/viz/tests/viz_basic_loads.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/viz/tests/viz_basic_loads.rs
use viz::scenario::{load_scenario_path, CreatureKind};

#[test]
fn viz_basic_scenario_loads_5_agents() {
    let scen = load_scenario_path("scenarios/viz_basic.toml").unwrap();
    assert_eq!(scen.agents.len(), 5);
    let wolves = scen.agents.iter().filter(|a| a.creature_type == CreatureKind::Wolf).count();
    let humans = scen.agents.iter().filter(|a| a.creature_type == CreatureKind::Human).count();
    assert_eq!(wolves, 1);
    assert_eq!(humans, 4);
}

#[test]
fn wolf_is_in_the_middle_of_the_humans() {
    let scen = load_scenario_path("scenarios/viz_basic.toml").unwrap();
    let wolf = scen.agents.iter().find(|a| a.creature_type == CreatureKind::Wolf).unwrap();
    // Humans form a square; wolf at the centroid (0, 0, 0).
    assert_eq!(wolf.pos, [0.0, 0.0, 0.0]);
}
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cargo test -p viz --test viz_basic_loads`
Expected: FAIL — file not found.

- [ ] **Step 3: Write `crates/viz/scenarios/viz_basic.toml`**

```toml
# Four humans in a 10-meter square, one wolf at the center.
# Run from the crate root: `cargo run -p viz -- scenarios/viz_basic.toml`.
# Expected visible behavior after a few hundred ticks: the wolf attacks the
# nearest human (UtilityBackend's Attack-within-range rule), HP drops, one
# human dies, a black death marker appears.

[[agent]]
creature_type = "Human"
pos = [-5.0, 0.0, -5.0]
hp = 100.0

[[agent]]
creature_type = "Human"
pos = [ 5.0, 0.0, -5.0]
hp = 100.0

[[agent]]
creature_type = "Human"
pos = [-5.0, 0.0,  5.0]
hp = 100.0

[[agent]]
creature_type = "Human"
pos = [ 5.0, 0.0,  5.0]
hp = 100.0

[[agent]]
creature_type = "Wolf"
pos = [ 0.0, 0.0,  0.0]
hp = 80.0
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `cargo test -p viz --test viz_basic_loads`
Expected: both tests PASS.

- [ ] **Step 5: Manual verification**

Run: `cargo run -p viz -- crates/viz/scenarios/viz_basic.toml`
Expected: window opens, stdout emits `[viz] tick=... alive=5 speed=1.00x wall=...s` lines approximately once per second; closing the window exits cleanly. (No pixels yet — the renderer is blank pending Plan 3.1, but the state is advancing, verifiable via stdout.)

Press `Space` to pause (HUD shows `[PAUSED]`); press `.` to single-step; press `R` to reset; press `[` / `]` to change speed. Each keypress should reflect in the next HUD line.

- [ ] **Step 6: Commit**

```bash
git add crates/viz/scenarios/viz_basic.toml crates/viz/tests/viz_basic_loads.rs
git commit -m "feat(viz): viz_basic scenario (4 humans + 1 wolf) for interactive use (Task 7/8)"
```

---

## Task 8: Visual-check checklist in `docs/engine/status.md`

**Files:**
- Modify (or create if missing): `docs/engine/status.md`

- [ ] **Step 1: Check whether `docs/engine/status.md` exists**

Run: `ls docs/engine/status.md`
If missing: proceed to Step 2 to create a minimal file. If present: proceed to Step 3 to append a section.

- [ ] **Step 2: If missing, create `docs/engine/status.md` with a skeleton**

```markdown
# Engine Status

A running tally of what's implemented, what's visually verifiable, and what
is still untouched. Companion to `spec.md` (authoritative requirements) and
`README.md` (orientation).

Last updated: 2026-04-19 (Plan 3.0 landed).

## Visual-check checklist

See below.
```

- [ ] **Step 3: Append the Visual-check checklist section**

Append (or replace if it already exists) this section in `docs/engine/status.md`:

```markdown
## Visual-check checklist

These are the eyeball tests a human runs against `cargo run -p viz --
crates/viz/scenarios/viz_basic.toml`. Each item is "I can see X when Y
happens" — they exist to catch regressions that unit tests might miss
because the same author writes both test and impl.

Unchecked items still need a visual confirmation after Plan 3.0 lands:

- [ ] **Agents appear as voxel markers at spawn positions.** Four white
  markers in a square, one grey marker in the centre. (Blocked until
  Plan 3.1 wires a voxel_engine renderer through the `App` loop — today
  the window is blank; use `cargo test -p viz --test agent_rendering`
  for the shadow-buffer equivalent.)
- [ ] **Wolf attacks nearest human when within 2 m range.** After ~300
  ticks the wolf reaches a human; red (MAT_ATTACK) overlay cells appear
  on the midpoint between attacker and target; the target's HP (visible
  via future HUD) drops. TTL: overlay fades after 5 ticks.
- [ ] **HP reaching zero produces a black death marker** that **persists
  for the rest of the session**, not just 5 ticks.
- [ ] **Hunger drops over time** (future cascade handler; not wired in
  Plan 3.0 — flag if `agent_hunger` decreases during `step`).
- [ ] **Eat micro restores hunger.** After an `AgentAte` event, the
  target's hunger field is observably higher than the pre-event value
  (delta ≤ `EAT_RESTORE` = 0.25).
- [ ] **Flee moves target away from threat.** `AgentFled` events should
  move the fleeing agent's marker one cell away from the pursuer each
  tick.
- [ ] **Announce produces an expanding marker ring** around the speaker
  at radii 1 m / 3 m / 5 m, each lasting ~5 ticks with staggered fade.
- [ ] **Reset (R key) clears all markers and restores spawn positions.**
- [ ] **Pause (Space) freezes state.** HUD wall-clock continues; tick
  counter does not advance.
- [ ] **Step (Period while paused) advances exactly one sim tick.**
- [ ] **Speed controls (`[` / `]`) halve and double real-time → sim-time
  conversion.** At 16x, HUD reports ~160 ticks/s; at 0.0625x, ~0.6 ticks/s.

### Known limitations (Plan 3.0 MVP)

- **Window is blank.** voxel_engine does not yet expose `run_app(cfg, app)`
  that wires the `App` trait to a Vulkan renderer; our driver only
  constructs a headless scene. Use `cargo test -p viz` to verify voxel
  writes via the `shadow_get` accessor, or wait for Plan 3.1.
- **No get_voxel on `Scene`.** We shadow writes in `AgentRenderer::shadow`
  for test assertions; see Plan 3.1 for a voxel_engine API addition.
- **Agents are single-voxel markers**, not sprites / creature models.
  Swap in Plan 3.1.
```

- [ ] **Step 4: Commit**

```bash
git add docs/engine/status.md
git commit -m "docs(viz): visual-check checklist for Plan 3.0 viz harness (Task 8/8)"
```

---

## Self-review notes

- **Spec coverage:** every item in the user's suggested 8-task breakdown is covered by Tasks 1–8.
- **voxel_engine API gaps** (flagged to the caller in the final report):
  1. No `voxel_engine::app::run_app(cfg, app)` — we roll our own winit loop.
  2. No `Scene::get_voxel(handle, cell)` — we shadow writes in `AgentRenderer::shadow`.
  3. No non-voxel entity rendering — agents rendered as single-voxel markers.
  4. `App` trait is defined but nothing in voxel_engine drives it; hitting `setup`/`tick`/`on_input` is our responsibility (done in `main.rs::Driver`).
  5. `VoxelRenderer` exists but expects `VulkanContext` + cameras; wiring it is Plan 3.1.
- **Determinism:** all sim state flows through `engine::step::step` with `UtilityBackend` (deterministic). The renderer reads state but never feeds randomness back — safe.
- **Test isolation:** every behavioural test uses `Scene::new_headless` (no Vulkan); tests run on CI without a GPU.
- **Frequent commits:** 8 tasks → 8 commits, each one produces a passing test suite.
